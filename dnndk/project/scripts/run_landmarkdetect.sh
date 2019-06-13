gst-launch-1.0 \
    xlnxvideosrc src-type=hdmi ! \
    "video/x-raw, width=640, height=480, format=UYVY" ! \
    videoconvert ! \
    "video/x-raw, width=640, height=480, format=BGR" ! \
    sdxlandmarkdetect ! \
    fpsdisplaysink video-sink=" kmssink sync=false plane-id=29 bus-id="b00c0000.v_mix" render-rectangle=\"<0,0,640,480>\" " text-overlay=true sync=false
