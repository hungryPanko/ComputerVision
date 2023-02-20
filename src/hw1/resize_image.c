#include <math.h>
#include "image.h"
	
float nn_interpolate(image im, float x, float y, int c){
	//find the nearest neighborhood
	int i,j;
	if( x<0 && y<0 ){
		i = 0;
		j = 0;
	}else if( x<0 ){
		i = 0;
		j = round(y);
	}else if( y<0 ){
		i = round(x);
		j = 0;
	}else{
		i = round(x);
		j = round(y);
	}
	int pixel_index = i + im.w*j + im.w*im.h*c; 
	float pixel_value = im.data[pixel_index];

	return pixel_value;
}

image nn_resize(image im, int w, int h){
	image dst_im = make_image(w, h, im.c);
	
	// Match up coordinates, System of equations
	//a*-.5 + b = -.5
	//a*(dst_im.w-0.5) + b = im.w
	float a_x = (float)im.w/(float)dst_im.w; 
	float b_x = 0.5*(a_x-1);
	float a_y = (float)im.h/(float)dst_im.h;
	float b_y = 0.5*(a_y-1);

	int i,j,k;
	for(k=0; k<dst_im.c; ++k){
		for(j=0; j<dst_im.h; ++j){
			for(i=0; i<dst_im.w; ++i){
				int pixel_index = i + dst_im.w*j + dst_im.w*dst_im.h*k; 
				//Map to old coords
				float x = a_x*(float)i + b_x;
				float y = a_y*(float)j + b_y;
				float pixel_value = nn_interpolate(im, x, y, k); 
				dst_im.data[pixel_index] = pixel_value;
			}
		}
	}
	return dst_im;
}


float bilinear_interpolate(image im, float x, float y, int c)
{
	//On outer edges use padding
	if( x < 0 && y < 0){ //top left corner
		return im.data[0 + im.w*im.h*c];
	}else if( x>im.w-1 && y<0 ){ //top right corner
		return im.data[(im.w-1) + im.w*im.h*c]; 
	}else if( x<0 && y>im.h-1 ){ //bottom left corner
		return im.data[ im.w*(im.h-1) + im.w*im.h*c]; 
	}else if( x>im.w-1 && y>im.h-1 ){ //last index
		return im.data[(im.w-1) + im.w*(im.h-1) + im.w*im.h*c]; 
	}else if( x < 0 ){ //left end
		int p2_x = 0;
		int p2_y = floor(y);
		int p4_x = 0; 
		int p4_y = floor(y)+1;
		float dist1 = p4_y - y;
		float dist2 = y - p2_y;
		int p2_pixel_index = p2_x + im.w*p2_y + im.w*im.h*c; 
		int p4_pixel_index = p4_x + im.w*p4_y + im.w*im.h*c; 
		float pixel_value = im.data[p2_pixel_index]*dist1 + im.data[p4_pixel_index]*dist2; 
		return pixel_value;
	}else if( y < 0 ){ // top end
		int p3_x = floor(x);
		int p3_y = 0; 
		int p4_x = floor(x)+1; 
		int p4_y = 0;
		float dist1 = p4_x - x;
		float dist2 = x - p3_x;
		int p3_pixel_index = p3_x + im.w*p3_y + im.w*im.h*c; 
		int p4_pixel_index = p4_x + im.w*p4_y + im.w*im.h*c; 
		float pixel_value = im.data[p3_pixel_index]*dist1 + im.data[p4_pixel_index]*dist2; 
		return pixel_value;
	}else if(x>im.w-1){ //right end
		int p1_x = floor(x);
		int p1_y = floor(y);
		int p3_x = floor(x);
		int p3_y = floor(y)+1; 
		float dist1 = p3_y - y;
		float dist2 = y - p1_y;
		int p1_pixel_index = p1_x + im.w*p1_y + im.w*im.h*c; 
		int p3_pixel_index = p3_x + im.w*p3_y + im.w*im.h*c; 
		float pixel_value = im.data[p1_pixel_index]*dist1 + im.data[p3_pixel_index]*dist2; 
		return pixel_value;
	}else if(y>im.h-1){ //bottom end
		int p1_x = floor(x);
		int p1_y = floor(y);
		int p2_x = floor(x)+1;
		int p2_y = floor(y);
		float dist1 = p2_x - x;
		float dist2 = x - p1_x;
		int p1_pixel_index = p1_x + im.w*p1_y + im.w*im.h*c; 
		int p2_pixel_index = p2_x + im.w*p2_y + im.w*im.h*c; 
		float pixel_value = im.data[p1_pixel_index]*dist1 + im.data[p2_pixel_index]*dist2; 
		return pixel_value;
	}
	else{ //middle pixels
		//p1: floor(x), floor(y), p2: floor(x+)+1, floor(y), p3: floor(x), floor(y)+1, p4:floor(x)+1, floor(y)+1
		// ex) x=0.7 y=1.2 --> (0,1) (1,2), (0,2), (1,2)
		int p1_x = floor(x);
		int p1_y = floor(y);
		int p2_x = floor(x)+1;
		int p2_y = floor(y);
		int p3_x = floor(x);
		int p3_y = floor(y)+1; 
		int p4_x = floor(x)+1; 
		int p4_y = floor(y)+1;
 	
		//q1 : dst1=floor(y)+1 - y, dst2= y - floor(y)
		float dist1 = p3_y - y;
		float dist2 = y - p1_y;

		//q1_r: dst1* p1.r + dst2*p3.r
		int p1_pixel_index = p1_x + im.w*p1_y + im.w*im.h*c; 
		int p3_pixel_index = p3_x + im.w*p3_y + im.w*im.h*c; 
		float q1_pixel_value = im.data[p1_pixel_index]*dist1 + im.data[p3_pixel_index]*dist2; 

		//q2_r: dst1* p2.r + dst2*p4.r
		int p2_pixel_index = p2_x + im.w*p2_y + im.w*im.h*c; 
		int p4_pixel_index = p4_x + im.w*p4_y + im.w*im.h*c; 
		float q2_pixel_value = im.data[p2_pixel_index]*dist1 + im.data[p4_pixel_index]*dist2; 
	
		//q1 : dst1=floor(y)+1 - y, dst2= y - floor(y)
		float dist3 = p2_x - x;
		float dist4 = x - p1_x;
		float pixel_value = q1_pixel_value*dist3 + q2_pixel_value*dist4; 

	    return pixel_value;
	}
}

image bilinear_resize(image im, int w, int h)
{
	image dst_im = make_image(w, h, im.c);
	
	// Match up coordinates, System of equations
	//a*-.5 + b = -.5
	//a*(dst_im.w-0.5) + b = im.w
	float a_x = (float)im.w/(float)dst_im.w; 
	float b_x = 0.5*(a_x-1.0);
	float a_y = (float)im.h/(float)dst_im.h;
	float b_y = 0.5*(a_y-1.0);

	int i,j,k;
	for(k=0; k<dst_im.c; ++k){
		for(j=0; j<dst_im.h; ++j){
			for(i=0; i<dst_im.w; ++i){
				int pixel_index = i + dst_im.w*j + dst_im.w*dst_im.h*k; 
				//Map to old coords
				float x = a_x*(float)i + b_x;
				float y = a_y*(float)j + b_y;
				//printf("index:%d, i:%d j:%d, k:%d, x:%f, y:%f\n",pixel_index,i,j,k,x,y);
				float pixel_value = bilinear_interpolate(im, x, y, k); 
				dst_im.data[pixel_index] = pixel_value;
			}
		}
	}
	return dst_im;
}
