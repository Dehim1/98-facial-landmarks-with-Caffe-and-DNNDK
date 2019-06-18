#ifndef __BBOX_H__
#define __BBOX_H__

class BBox
{
private:
public:
	float m_x1;
	float m_y1;
	float m_x2;
	float m_y2;
	float m_xCenter;
	float m_yCenter;
	float m_width;
	float m_height;
public:
	static BBox FromBoundaries(float x1, float y1, float x2, float y2)
	{
		BBox bbox;
		bbox.m_x1 = x1;
		bbox.m_y1 = y1;
		bbox.m_x2 = x2;
		bbox.m_y2 = y2;
		bbox.m_xCenter = (x2+x1)/2;
		bbox.m_yCenter = (y2+y1)/2;
		bbox.m_width = x2-x1;
		bbox.m_height = y2-y1;
		return bbox;
	}

	static BBox FromCenterAndDimensions(float xCenter, float yCenter, float width, float height)
	{
		BBox bbox;
		bbox.m_x1 = xCenter - width/2;
		bbox.m_y1 = yCenter - height/2;
		bbox.m_x2 = bbox.m_x1 + width;
		bbox.m_y2 = bbox.m_y1 + height;
		bbox.m_xCenter = xCenter;
		bbox.m_yCenter = yCenter;
		bbox.m_width = width;
		bbox.m_height = height;
		return bbox;
	}
};

#endif //__BOX_H__
