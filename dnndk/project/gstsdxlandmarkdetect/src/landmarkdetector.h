#ifndef __LANDMARKDETECTOR_H__
#define __LANDMARKDETECTOR_H__

#include <opencv2/opencv.hpp>
#include "bbox.h"
#include "dnndk.h"

#ifdef _PERFORMANCE
#define _T(func)                                                          \
  {                                                                       \
    auto _start = system_clock::now();                                    \
    func;                                                                 \
    auto _end = system_clock::now();                                      \
    auto duration = (duration_cast<microseconds>(_end - _start)).count(); \
    string tmp = #func;                                                   \
    tmp = tmp.substr(0, tmp.find('('));                                   \
    std::cout << "[TimeMeasure]" << left << setw(30) << tmp;                   \
    std::cout << right << setw(10) << duration << " us" << std::endl;               \
  }
#else
#define _T(func) func;
#endif

#ifdef _DEBUG
#define __TRACE__ \
  { std::cout << "[TRACE]" << __FILE__ << " : " << __LINE__ << " " << __FUNCTION__ << std::endl; }
#else
#define __TRACE__
#endif

class LandmarkDetector
{
private:
	cv::Mat* m_srcImg;
	cv::Mat m_dpuImg;
	BBox* m_bbox;
	int m_nLandmarks;
	float* m_landmarks;

	std::string m_kernelName;
	std::string m_inputNode;
	std::string m_outputNode;
	DPUKernel* m_kernel;
	DPUTask* m_task;
public:


	void Init(std::string kernelName = "98_landmark", std::string inputNode = "Conv1", std::string outputNode = "Dense_98")
	{
		m_kernelName = kernelName;
		m_inputNode = inputNode;
		m_outputNode = outputNode;

		m_kernel = dpuLoadKernel(m_kernelName.c_str());
		m_task = dpuCreateTask(m_kernel, 0);
		m_nLandmarks = dpuGetOutputTensorChannel(m_task, m_outputNode.c_str())/2;
		m_landmarks = new float[2*m_nLandmarks];
		dpuDestroyTask(m_task);
	}

	void Finalize()
	{
		dpuDestroyTask(m_task);
		dpuDestroyKernel(m_kernel);
		delete[] m_landmarks;
	}

	void CreateTask()
	{
		m_task = dpuCreateTask(m_kernel, 0);
	}

	void Run(cv::Mat* img, BBox* bbox)
	{
		m_srcImg = img;
		m_bbox = bbox;
		ClipBBox();
		if (m_bbox->m_width > 0.0f && m_bbox->m_height > 0.0f)
		{
			CropImg();
			ResizeImg();
			DetectLandmarks();
			ProjectBBoxLandmarksToImg();
		}
	}

	std::pair<int, float*> GetLandmarks()
	{
		float* landmarks = new float[m_nLandmarks*2];
		for (int idx = 0; idx < m_nLandmarks*2; ++idx)
		{
			landmarks[idx] = m_landmarks[idx];
		}
		return std::pair<int, float*>(m_nLandmarks, landmarks);
	}

	void ClipBBox()
	{
		float width = m_srcImg->cols-1;
		float height = m_srcImg->rows-1;
		float x1 = floor(m_bbox->m_x1);
		float y1 = floor(m_bbox->m_y1);
		float x2 = ceil(m_bbox->m_x2);
		float y2 = ceil(m_bbox->m_y2);
		x1 = std::min(std::max(x1, 0.0f), width);
		y1 = std::min(std::max(y1, 0.0f), height);
		x2 = std::min(std::max(x2, 0.0f), width);
		y2 = std::min(std::max(y2, 0.0f), height);
		*m_bbox = BBox::FromBoundaries(x1, y1, x2, y2);
	}

	void CropImg()
	{
		cv::Rect roi = cv::Rect(cv::Point(m_bbox->m_x1, m_bbox->m_y1), cv::Point(m_bbox->m_x2, m_bbox->m_y2));
		m_dpuImg = (*m_srcImg)(roi);
	}

	void ResizeImg(int interpolation=cv::INTER_NEAREST)
	{
		int width = dpuGetInputTensorWidth(m_task, m_inputNode.c_str());
		int height = dpuGetInputTensorHeight(m_task, m_inputNode.c_str());
		cv::Size size = cv::Size(width, height);
		cv::resize(m_dpuImg, m_dpuImg, size, 0, 0, interpolation);
	}

	void DetectLandmarks()
	{
		_T(dpuSetInputImage2(m_task, m_inputNode.c_str(), m_dpuImg));
		_T(dpuRunTask(m_task));
		dpuGetOutputTensorInHWCFP32(m_task, m_outputNode.c_str(), m_landmarks, 2*m_nLandmarks);
	}

	void ProjectBBoxLandmarksToImg()
	{
		for (int idx = 0; idx < m_nLandmarks; ++idx)
		{
			m_landmarks[2*idx] = m_landmarks[2*idx]*m_bbox->m_width;
			m_landmarks[2*idx+1] = m_landmarks[2*idx+1]*m_bbox->m_height;
			m_landmarks[2*idx] = m_landmarks[2*idx]+m_bbox->m_xCenter;
			m_landmarks[2*idx+1] = m_landmarks[2*idx+1]+m_bbox->m_yCenter;
		}
	}

	void DrawBBox(const cv::Scalar& color = cv::Scalar(0, 255, 0, 255), int thickness=1)
	{
		cv::rectangle(*m_srcImg, cv::Point(m_bbox->m_x1, m_bbox->m_y1), cv::Point(m_bbox->m_x2, m_bbox->m_y2), color, thickness);
	}

	void DrawLandmarks(const cv::Scalar& color = cv::Scalar(0, 0, 255, 255), int thickness=1)
	{
		for(int idx = 0; idx < m_nLandmarks; ++idx)
		{
			cv::circle(*m_srcImg, cv::Point(m_landmarks[2*idx], m_landmarks[2*idx+1]), 1, color, thickness);
		}
	}
};
#endif //__LANDMARKDETECTOR_H__
