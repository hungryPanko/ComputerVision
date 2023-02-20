#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include "image.h"
#define DEBUG_MODE 0
#define debug_print(fmt, ...) \
            do { if (DEBUG_MODE) fprintf(stderr, fmt, __VA_ARGS__); } while (0)

float get_pixel(image im, int x, int y, int c)
{
	debug_print("++++++++++ %s +++++++++\n", __func__);
	debug_print("Image info: height:%d width:%d \n", im.h, im.w);

	// Padding. ex) Indexing '-20' retrun 0. Indexing '300' for 255 image, return 255-1
	x = (x<0) ? 0 : x;
	y = (y<0) ? 0 : y;
	x = (x>=im.w) ? (im.w-1) : x;
	y = (y>=im.h) ? (im.h-1) : y;
	debug_print("x: %d y:%d\n", x, y);
	
	// Find the pixel index
	int pixel_index = x + y*im.w + c*im.w*im.h; 
	debug_print("pixel_index: %d\n", pixel_index); 

	// Return pixel value
	return im.data[pixel_index];
}

void set_pixel(image im, int x, int y, int c, float v)
{
   // Bound checking. Do nothing on invalid x or y.  
	debug_print("++++++++++ %s +++++++++\n", __func__);
	if( x<0 || x>=im.w || y<0 || y>=im.h){
		debug_print("x: %d y:%d\n", x, y);
		return;	
	}
	
	// Find the pixel index
	int pixel_index = x + y*im.w + c*im.w*im.h; 
	debug_print("pixel_index: %d\n", pixel_index); 
	
	// Set pixel on image 
	im.data[pixel_index] = v;
	return;
}

image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
	//indexing each pixel in source image and fill the pixel value in dest image. 
	int i,j,k;
	for(k=0; k<im.c; ++k){
		for(j=0; j<im.h; ++j){
			for(i=0; i<im.w; ++i){
				int pixel_index = i + im.w*j + im.w*im.h*k;
				copy.data[pixel_index] = (float)im.data[pixel_index];
			}
		}
	}

    return copy;
}

image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    image gray = make_image(im.w, im.h, 1);
	
	debug_print("++++++++++ %s +++++++++\n", __func__);
	debug_print("Image info: height:%d width:%d \n", im.h, im.w);
	
	// indexing each pixel in source image 
	int i,j;
	for(j=0; j<im.h; ++j){
		for(i=0; i<im.w; ++i){
			// Index for R,G,B from src image
			int index_r = i + im.w*j; 
			int index_g = i + im.w*j + im.w*im.h*1;
			int index_b = i + im.w*j + im.w*im.h*2;
			
			//Y' = 0.299 R' + 0.587 G' + .114 B'
			float pixel_value = 0.299*im.data[index_r] + 0.587*im.data[index_g] + 0.114*im.data[index_b];

			//Save on grey image
			int pixel_index = i + im.w*j;
			gray.data[pixel_index] = pixel_value;
		}	
	}
    return gray;
}

void shift_image(image im, int c, float v)
{
	debug_print("++++++++++ %s +++++++++\n", __func__);
	debug_print("Image info: height:%d width:%d \n", im.h, im.w);
	
	// indexing each pixel in source image 
	int i,j;
	for(j=0; j<im.h; ++j){
		for(i=0; i<im.w; ++i){
			int pixel_index = i + im.w*j + im.w*im.h*c; 
			//shifting pixel value by v. 
			im.data[pixel_index] = im.data[pixel_index]+v;  
		}
	}
	return;
}

void scale_image(image im, int c, float v)
{
	debug_print("++++++++++ %s +++++++++\n", __func__);
	debug_print("Image info: height:%d width:%d \n", im.h, im.w);
	
	// indexing each pixel in source image 
	int i,j;
	for(j=0; j<im.h; ++j){
		for(i=0; i<im.w; ++i){
			int pixel_index = i + im.w*j + im.w*im.h*c; 
			//scale up by v. 
			im.data[pixel_index] = im.data[pixel_index]*v;  
		}
	}
	return;
}

			
void clamp_image(image im)
{
    // TODO Fill this in
	debug_print("++++++++++ %s +++++++++\n", __func__);
	debug_print("Image info: height:%d width:%d \n", im.h, im.w);
	
	// indexing each pixel in source image 
	int i,j,k;
	for(k=0; k<im.c; ++k){
		for(j=0; j<im.h; ++j){
			for(i=0; i<im.w; ++i){
				float pixel_value = get_pixel(im, i, j, k); 

				// upper bound: 1.0, lowder bound: 0.0
				pixel_value = (pixel_value>1.0) ? 1.0 : pixel_value;
				pixel_value = (pixel_value<0.0) ? 0.0 : pixel_value;
				set_pixel(im, i,j,k, pixel_value);
			}
		}
	}
	return;
}


// These might be handy
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im)
{
	debug_print("++++++++++ %s +++++++++\n", __func__);
	debug_print("Image info: height:%d width:%d \n", im.h, im.w);
	// indexing each pixel in source image 
	int i,j;
	for(j=0; j<im.h; ++j){
		for(i=0; i<im.w; ++i){
			// Index for R,G,B from src image
			int index_r = i + im.w*j; 
			int index_g = i + im.w*j + im.w*im.h*1;
			int index_b = i + im.w*j + im.w*im.h*2;
			
			float H,S,V;
			float R = im.data[index_r];
			float G = im.data[index_g];
			float B = im.data[index_b];
			
			V = three_way_max(R, G, B); //V = max(R,G,B)
			float m = three_way_min(R, G, B); //m = min(R,G,B)
			float C = V - m; //C=v-m 
			S = (V==0) ? 0 : C/V; //if V=0, S=0, otherwise S=C/V			

			float H_prime=0;
			// if C=0, then H = undefined 
			if(C == 0.0){ 
				H = 0;  // To do: Original instruction H = 0;
			}
			else{ 
				if(V==R){ //if V=R, H' = (G-B)/C
					H_prime = (G-B) / C; 
				}else if(V==G){ //if V=G, H'=(B-R)/C + 2
					H_prime = (B-R) / C + 2; 
				}else if(V==B){ //if V=B, H'=(R-G)/C + 4
					H_prime = (R-G) / C + 4; 
				}else{
					printf("V is not matching to R,G,orB");
				}

				//if H'<0: H=H'/6+1, otherwise H=h'/6  
				H = (H_prime<0) ? (H_prime/6.0 + 1.0) : (H_prime/6.0);
			}
			
			//store HSV
			im.data[index_r] = H; 
			im.data[index_g] = S;
			im.data[index_b] = V;
		}
	}
}

void hsv_to_rgb(image im)
{
	debug_print("++++++++++ %s +++++++++\n", __func__);
	debug_print("Image info: height:%d width:%d \n", im.h, im.w);

	// indexing each pixel in source image 
	int i,j;
	for(j=0; j<im.h; ++j){
		for(i=0; i<im.w; ++i){
			// Index for H,S,V from src image
			int index_h = i + im.w*j; 
			int index_s = i + im.w*j + im.w*im.h*1;
			int index_v = i + im.w*j + im.w*im.h*2;
			float H = im.data[index_h];
			float S = im.data[index_s];
			float V = im.data[index_v];
			
			// C=S*V
			float C = S*V; 
			// H'=6H
			float H_prime = (6.0*H);
			// X=C(1-| H'mod2 - 1 |)
			float X = C*(1-fabs(fmodf(H_prime,2.0f)-1));	

			float R1=0;
			float G1=0;
			float B1=0;
			if(H_prime >= 0 && H_prime < 1){
				R1=C;
				G1=X;
				B1=0;
			}else if(H_prime < 2){
				R1=X;
				G1=C;
				B1=0;
			}else if(H_prime < 3){
				R1=0;
				G1=C;
				B1=X;
			}else if(H_prime < 4){
				R1=0;
				G1=X;
				B1=C;
			}else if(H_prime < 5){
				R1=X;
				G1=0;
				B1=C;
			}else if(H_prime < 6){
				R1=C;
				G1=0;
				B1=X;
			}else{
				//printf("H_prime is out of range [0,6)");
				if(H==0){
						R1=0;
						G1=0;
						B1=0;
				}
			}
			
			float m = V - C; //m=V-C
			im.data[index_h] = R1 + m;
			im.data[index_s] = G1 + m;
			im.data[index_v] = B1 + m;
		}
	}
}




