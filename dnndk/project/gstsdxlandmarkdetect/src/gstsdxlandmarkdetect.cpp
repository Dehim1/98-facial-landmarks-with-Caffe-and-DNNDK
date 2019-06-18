/*
-- (c) Copyright 2018 Xilinx, Inc. All rights reserved.
--
-- This file contains confidential and proprietary information
-- of Xilinx, Inc. and is protected under U.S. and
-- international copyright and other intellectual property
-- laws.
--
-- DISCLAIMER
-- This disclaimer is not a license and does not grant any
-- rights to the materials distributed herewith. Except as
-- otherwise provided in a valid license issued to you by
-- Xilinx, and to the maximum extent permitted by applicable
-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
-- WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
-- BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and

-- (2) Xilinx shall not be liable (whether in contract or tort,
-- including negligence, or under any other theory of
-- liability) for any loss or damage of any kind or nature
-- related to, arising under or in connection with these
-- materials, including for any direct, or any indirect,
-- special, incidental, or consequential loss or damage
-- (including loss of data, profits, goodwill, or any type of
-- loss or damage suffered as a result of any action brought
-- by a third party) even if such damage or loss was
-- reasonably foreseeable or Xilinx had been advised of the
-- possibility of the same.
--
-- CRITICAL APPLICATIONS
-- Xilinx products are not designed or intended to be fail-
-- safe, or for use in any application requiring fail-safe
-- performance, such as life-support or safety devices or
-- systems, Class III medical devices, nuclear facilities,
-- applications related to the deployment of airbags, or any
-- other applications that could lead to death, personal
-- injury, or severe property or environmental damage
-- (individually and collectively, "Critical
-- Applications"). Customer assumes the sole risk and
-- liability of any use of Xilinx products in Critical
-- Applications, subject only to applicable laws and
-- regulations governing limitations on product liability.
--
-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
-- PART OF THIS FILE AT ALL TIMES.
*/

/**
 * SECTION:sdxlandmark_detect
 *
 * This is an example SDX accelerated landmarkdetect.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v videotestsrc ! sdxlandmarkdetect ! kmssink
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <chrono>
#include <vector>
/* Header files for OpenCV */
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include "bbox.h"
#include "landmarkdetector.h"
#include "densebox.h"
#include "gstsdxlandmarkdetect.h"

using namespace std;
using namespace cv;
using namespace std::chrono;
using namespace deephi;

GST_DEBUG_CATEGORY_STATIC(gst_sdx_landmark_detect_debug);
#define GST_CAT_DEFAULT gst_sdx_landmark_detect_debug

#define SDX_LANDMARK_DETECT_INPUT_PER_OUTPUT 1

#define SDX_LANDMARK_DETECT_CAPS    \
  "video/x-raw, "               \
  "format = (string) {BGR}, "   \
  "width = (int) [ 1, 3840 ], "  \
  "height = (int) [ 1, 2160 ], " \
  "framerate = " GST_VIDEO_FPS_RANGE

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS, GST_STATIC_CAPS(SDX_LANDMARK_DETECT_CAPS));

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS(SDX_LANDMARK_DETECT_CAPS));

#define gst_sdx_landmark_detect_parent_class parent_class
G_DEFINE_TYPE(GstSdxLandmarkDetect, gst_sdx_landmark_detect, GST_TYPE_SDX_BASE);

#define GST_SDX_LANDMARK_DETECT_GET_PRIVATE(obj) \
  (G_TYPE_INSTANCE_GET_PRIVATE((obj), GST_TYPE_SDX_LANDMARK_DETECT, GstSdxLandmarkDetectPrivate))

struct _GstSdxLandmarkDetectPrivate {
  DenseBox densebox;
  LandmarkDetector landmarkdetector;
};

static GstFlowReturn gst_sdx_landmark_detect_process_frames_ip(GstSdxBase *base, GstSdxFrame *frame) {
  GstSdxLandmarkDetect *filter = GST_SDX_LANDMARK_DETECT(base);
  GstSdxLandmarkDetectPrivate *p = filter->priv;
  GstVideoInfo *info = NULL;

  g_return_val_if_fail(frame != NULL, GST_FLOW_ERROR);

  info = &frame->info;

  thread_local static DPUTask *task;
  thread_local static int init_switch = 1;
  if (init_switch == 1) {
    init_switch = 0;
    task = dpuCreateTask(p->densebox.kernel, 0);
    p->densebox.InsertTask(task);
    p->landmarkdetector.CreateTask();
  }

  /* create Mat image from input buffer to send to DPU */
  Mat image(GST_VIDEO_INFO_HEIGHT(info), GST_VIDEO_INFO_WIDTH(info), CV_8UC3,
            GST_VIDEO_FRAME_PLANE_DATA(&frame->vframe, 0),
            GST_VIDEO_FRAME_PLANE_STRIDE(&frame->vframe, 0));

  vector<array<float, 5>> faceDetectResult;
  p->densebox.Run(task, image, &faceDetectResult);

  for (auto iter = faceDetectResult.begin(); iter != faceDetectResult.end(); ++iter)
  {
	  BBox bbox = BBox::FromBoundaries((*iter)[0], (*iter)[1], (*iter)[2], (*iter)[3]);
	  p->landmarkdetector.Run(&image, &bbox);
	  p->landmarkdetector.DrawBBox();
	  p->landmarkdetector.DrawLandmarks();
  }


  return GST_FLOW_OK;
}

static gboolean gst_sdx_landmark_detect_start(GstBaseTransform *trans) {
  GstSdxLandmarkDetect *filter = GST_SDX_LANDMARK_DETECT(trans);

  filter->priv = GST_SDX_LANDMARK_DETECT_GET_PRIVATE(filter);
  filter->priv->densebox.Init();
  filter->priv->landmarkdetector.Init();
  return TRUE;
}

static gboolean gst_sdx_landmark_detect_stop(GstBaseTransform *trans) {
  GstSdxLandmarkDetect *filter = GST_SDX_LANDMARK_DETECT(trans);
  filter->priv->densebox.Finalize();
  filter->priv->landmarkdetector.Finalize();
  return TRUE;
}

static void gst_sdx_landmark_detect_finalize(GObject *object) {
  GstSdxLandmarkDetect *filter = GST_SDX_LANDMARK_DETECT(object);
  int iret = dpuClose();
  if (iret) {
    GST_ERROR_OBJECT(filter, "failed to close DPU device. error : %d", iret);
  }
  GST_INFO_OBJECT(filter, "Successfully close DPU");

  G_OBJECT_CLASS(parent_class)->finalize(object);
}

static void gst_sdx_landmark_detect_init(GstSdxLandmarkDetect *filter) {
  GstSdxBase *base = GST_SDX_BASE(filter);
  int iret = dpuOpen();
  if (iret) {
    GST_ERROR_OBJECT(filter, "failed to open DPU device. error : %d", iret);
  }
  GST_INFO_OBJECT(filter, "Successfully open DPU");

  gst_sdx_base_set_inputs_per_output(base, SDX_LANDMARK_DETECT_INPUT_PER_OUTPUT);
}

static void gst_sdx_landmark_detect_class_init(GstSdxLandmarkDetectClass *klass) {
  GObjectClass *gobject_class;
  GstElementClass *element_class;
  GstBaseTransformClass *transform_class;
  GstSdxBaseClass *sdxbase_class;

  g_type_class_add_private(klass, sizeof(GstSdxLandmarkDetectPrivate));

  gobject_class = G_OBJECT_CLASS(klass);
  element_class = GST_ELEMENT_CLASS(klass);
  transform_class = GST_BASE_TRANSFORM_CLASS(klass);
  sdxbase_class = GST_SDX_BASE_CLASS(klass);

  gst_element_class_add_static_pad_template(element_class, &sink_template);
  gst_element_class_add_static_pad_template(element_class, &src_template);

  gobject_class->finalize = gst_sdx_landmark_detect_finalize;

  transform_class->start = GST_DEBUG_FUNCPTR(gst_sdx_landmark_detect_start);
  transform_class->stop = GST_DEBUG_FUNCPTR(gst_sdx_landmark_detect_stop);

  sdxbase_class->map_flags = GST_MAP_READWRITE;
  sdxbase_class->process_frame_ip = GST_DEBUG_FUNCPTR(gst_sdx_landmark_detect_process_frames_ip);

  gst_element_class_set_static_metadata(element_class, "SDx Landmark Detection", "Filter/Effect/Video",
                                        "Detects landmarks using DNN",
                                        "Naveen Cherukuri <naveenc@xilinx.com>");
}

static gboolean plugin_init(GstPlugin *plugin) {
  GST_DEBUG_CATEGORY_INIT(gst_sdx_landmark_detect_debug, "sdxlandmarkdetect", 0, "SDx Landmark Detection");

  return gst_element_register(plugin, "sdxlandmarkdetect", GST_RANK_NONE, GST_TYPE_SDX_LANDMARK_DETECT);
}

/* PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "sdxlandmarkdetect"
#endif

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, sdxlandmarkdetect, "SDx Landmark Detection plugin",
                  plugin_init, "0.1", "LGPL", "GStreamer SDX", "http://xilinx.com/")
