#!/bin/sh

# Note: modify the device property of v4l2src plugin according to your system 
#       which you can check with command "video_cmd -S"

gst-launch-1.0 \
    xlnxvideosrc src-type="hdmi" ! \
    "video/x-raw, width=1920, height=1080, format=UYVY" ! \
    videoconvert ! \
    "video/x-raw, width=1920, height=1080, format=BGR" ! \
    sdxlandmarkdetect ! \
    videoscale method=0 add-borders=false ! \
    video/x-raw,width=1792,height=1008 ! \
    fpsdisplaysink video-sink=" kmssink sync=false plane-id=29 bus-id="b00c0000.v_mix" render-rectangle=\"<0,0,1792,1008>\" " text-overlay=true sync=false
