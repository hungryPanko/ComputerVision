#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#define TWOPI 6.2831853

void l1_normalize(image im)
{
	int i,j; 
	float v_ttl=0; 
	for(j=0; j<im.h; ++j){
		for(i=0; i<im.w; ++i){
			v_ttl += get_pixel(im, i, j, 0);
		}
	}
	float norm = v_ttl/im.h*im.w; // the kernl: sum to 1
	
	for(j=0; j<im.h; ++j){
		for(i=0; i<im.w; ++i){
			float v = get_pixel(im, i, j, 0);
			set_pixel(im, i, j, 0, v/norm);  // divide by norm
		}
	}
}

// Box Filter
// | 1 1 1 | 
// | 1 1 1 |  * 1/N*M
// | 1 1 1 | 
image make_box_filter(int w)
{
    image im = make_image(w,w,1); // one channel 
	
	int i,j; 
	for(j=0; j<im.h; ++j){
		for(i=0; i<im.w; ++i){
			set_pixel(im, i, j, 0, 1); // fill kernel with all 1s
		}
	}
	l1_normalize(im); // sum to 1
	return im;
}


// For this function we have a few scenarios. 
// With normal convolutions we do a weighted sum over an area of the image.
// With multiple channels in the input image there are a few possible cases we want to handle:
image convolve_image(image im, image filter, int preserve)
{
	if(im.c == 1){ //if image channel is 1, convolution with filter channel0 --> output has 1 channel. 
		image im_result = make_image(im.w, im.h, 1);
		int i,j,l,m;
		for(j=0; j<im.h; ++j){
			for(i=0; i<im.w; ++i){
				float v = 0;
				for(l=-1*(int)floor(filter.h/2); l<= (int)floor(filter.h/2); l++){ // filter 1 layer
					for(m=-1*(int)floor(filter.w/2); m <= (int)floor(filter.w/2); m++){
						float v_im = get_pixel(im, i+m, j+l, 0);
						float v_filter = get_pixel(filter, m+(int)floor(filter.w/2), l+(int)floor(filter.h/2), 0);
						v += v_im*v_filter;
					}
				}
				set_pixel(im_result, i, j, 0, v); 
			}
		}
		return im_result;
	}else{ //if image channels are more than 2
		assert(im.c >= 2);
		if(preserve == 1){ // return same number of channel, ex) 3 channels input --> 3 channels output
			image im_result = make_image(im.w, im.h, im.c);
			int i,j,k,l,m;
			for(k=0; k<im.c; ++k){ 
				for(j=0; j<im.h; ++j){
					for(i=0; i<im.w; ++i){
						float v = 0;
						for(l=-1*(int)floor(filter.h/2); l<= (int)floor(filter.h/2); l++){ // filter 1 layer
							for(m=-1*(int)floor(filter.w/2); m <= (int)floor(filter.w/2); m++){
								float v_im = get_pixel(im, i+m, j+l, k);
								float v_filter = 0;
								if( filter.c == 1){ // if filter has 1 channel
									v_filter = get_pixel(filter, m+(int)floor(filter.w/2), l+(int)floor(filter.h/2), 0);
								}else{ // if image and filter have the same number of channel
									assert(im.c == filter.c);
									v_filter = get_pixel(filter, m+(int)floor(filter.w/2), l+(int)floor(filter.h/2), k);
								}
								v += v_im*v_filter;
							}
						}
						set_pixel(im_result, i, j, k, v); 
					}
				}
			}
			return im_result;	
		}else{ //preserve==0, return 1 channer img, ex) 3 channels intput --> 1 channels output
			image im_result = make_image(im.w, im.h, 1);
			int i,j,k,l,m;
			for(j=0; j<im.h; ++j){
				for(i=0; i<im.w; ++i){
					float v = 0;
					for(k=0; k<im.c; ++k){ 
						float v_channel = 0;
						for(l=-1*(int)floor(filter.h/2); l<= (int)floor(filter.h/2); l++){ // filter 1 layer
							for(m=-1*(int)floor(filter.w/2); m <= (int)floor(filter.w/2); m++){
								float v_im = get_pixel(im, i+m, j+l, k);
								float v_filter = 0;
								if( filter.c == 1 ){
									v_filter = get_pixel(filter, m+(int)floor(filter.w/2), l+(int)floor(filter.h/2), 0);
								}
								else{ // if image and filter have the same number of channel
									assert(im.c == filter.c);
									v_filter = get_pixel(filter, m+(int)floor(filter.w/2), l+(int)floor(filter.h/2), k);
								}
								v_channel += v_im*v_filter;
							}
						}
						v += v_channel;
					}
					set_pixel(im_result, i, j, 0, v); 
				}
			}	
			return im_result;
		}
	}
}

// highpass filter
// | 0 -1  0 |
// |-1  4 -1 |
// | 0 -1  0 |
image make_highpass_filter()
{
    image im = make_image(3,3,1);
	
	set_pixel(im, 0, 0, 0, 0);
	set_pixel(im, 1, 0, 0, -1);
	set_pixel(im, 2, 0, 0, 0);
	set_pixel(im, 0, 1, 0, -1);
	set_pixel(im, 1, 1, 0, 4);
	set_pixel(im, 2, 1, 0, -1);
	set_pixel(im, 0, 2, 0, 0);
	set_pixel(im, 1, 2, 0, -1);
	set_pixel(im, 2, 2, 0, 0);
	return im;
}

// Sharpen filter
// | 0 -1  0 |
// |-1  5 -1 |
// | 0 -1  0 |
image make_sharpen_filter()
{
    image im = make_image(3,3,1);
	
	set_pixel(im, 0, 0, 0, 0);
	set_pixel(im, 1, 0, 0, -1);
	set_pixel(im, 2, 0, 0, 0);
	set_pixel(im, 0, 1, 0, -1);
	set_pixel(im, 1, 1, 0, 5);
	set_pixel(im, 2, 1, 0, -1);
	set_pixel(im, 0, 2, 0, 0);
	set_pixel(im, 1, 2, 0, -1);
	set_pixel(im, 2, 2, 0, 0);
    return im;
}

// Emboss filter
// |-2 -1  0 |
// |-1  1  1 |
// | 0  1  2 |
image make_emboss_filter()
{
    image im = make_image(3,3,1);
	
	set_pixel(im, 0, 0, 0, -2);
	set_pixel(im, 1, 0, 0, -1);
	set_pixel(im, 2, 0, 0, 0);
	set_pixel(im, 0, 1, 0, -1);
	set_pixel(im, 1, 1, 0, 1);
	set_pixel(im, 2, 1, 0, 1);
	set_pixel(im, 0, 2, 0, 0);
	set_pixel(im, 1, 2, 0, 1);
	set_pixel(im, 2, 2, 0, 2);
    return im;
}

// Question 2.2.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
// Answer: Highpass filter does not need to be preserve. Because it is sum to zero, that shows only gradient information. 
// The other filters which sum to '1' might be better to preserve, since it intends to keep original pixel values as well. 
// However, whether output filter is used for processing in later or just for final result might the usage of preserve. 

// Question 2.2.2: Do we have to do any post-processing for the above filters? Which ones and why?
// Answer: Highpass, Sharpen, and Emboss filter need post-processing such as clamp(). Because in some cases, the pixel values 
// became large negative or greater than highest pixel values. For example, if neighboring pixels are set to 1 for '-1' place in kernel,
// and 0 for 'postitive' value in kenel, the resulting value becomes large negative value. 


//filter that smooths using a gaussian with that sigma. 

image make_gaussian_filter(float sigma)
{
	// 99% of the probability mass for a gaussian is within +/- 3 standard deviations
	//  so make the kernel be 6 times the size of sigma.
 	// But also we want an odd number, so make it be the next highest odd integer from 6x sigma.
	int size = sigma*6+1;
    image im = make_image(size,size,1);
    
	int i,j; 
	for(j=-sigma*3; j<=sigma*3 ; ++j){
		for(i=-sigma*3; i<=sigma*3; ++i){
			float v = 1/(float)(TWOPI*sigma*sigma)*exp( -1*(float)(i*i+j*j)/(float)(2*sigma*sigma) );
			set_pixel(im, i+sigma*3, j+sigma*3, 0, v);
		}
	}
	l1_normalize(im);
    return im;
}

// Add two images
image add_image(image a, image b)
{
	//checks that the images are the same size
    assert(a.w == b.w);
    assert(a.h == b.h);
    
	image im = make_image(a.w, a.h, MAX(a.c, b.c));
	int i,j,k;
	for(k=0; k<im.c; ++k){ 
		for(j=0; j<im.h; ++j){
			for(i=0; i<im.w; ++i){
				float v;
				if(a.c == b.c){ // if both image has the same number of channel 
					v = get_pixel(a, i, j, k) + get_pixel(b, i, j, k);
				}else if(a.c > b.c){ // ex) if a image has 3, b image has 1 channel.
					v = get_pixel(a, i, j, k) + get_pixel(b, i, j, 0);
				}else{ // ex)if a image has 1 channel, b image has 3.
					v = get_pixel(a, i, j, 0) + get_pixel(b, i, j, k);
				}
				set_pixel(im, i, j, k, v);
			}
		}
	}
    return im;
}

// Subtract images (a-b)
image sub_image(image a, image b)
{
	//checks that the images are the same size
    assert(a.w == b.w);
    assert(a.h == b.h);
	
	image im = make_image(a.w, a.h, MAX(a.c, b.c));
	int i,j,k;
	for(k=0; k<im.c; ++k){ 
		for(j=0; j<im.h; ++j){
			for(i=0; i<im.w; ++i){
				float v;
				if(a.c == b.c){ // if both image has the same number of channel
					v = get_pixel(a, i, j, k) - get_pixel(b, i, j, k);
				}else if(a.c > b.c){ // ex) if a iamge has 3 channels, b image has 1 channel.
					v = get_pixel(a, i, j, k) - get_pixel(b, i, j, 0);
				}else{ // ex) if a image has 1 channel, b image has 3. 
					v = get_pixel(a, i, j, 0) - get_pixel(b, i, j, k);
				}
				set_pixel(im, i, j, k, v);
			}
		}
	}
    return im;
}

// Sobel filter - check gradient of X direction 
// |-1  0  1 |
// |-2  0  2 |
// |-1  0  1 |
image make_gx_filter()
{
    image im = make_image(3,3,1);
	
	set_pixel(im, 0, 0, 0, -1);
	set_pixel(im, 1, 0, 0, 0);
	set_pixel(im, 2, 0, 0, 1);
	set_pixel(im, 0, 1, 0, -2);
	set_pixel(im, 1, 1, 0, 0);
	set_pixel(im, 2, 1, 0, 2);
	set_pixel(im, 0, 2, 0, -1);
	set_pixel(im, 1, 2, 0, 0);
	set_pixel(im, 2, 2, 0, 1);
    return im;
}

// Sobel filter - check gradient of Y direction 
// |-1 -2 -1 |
// | 0  0  0 |
// | 1  2  1 |
image make_gy_filter()
{
    image im = make_image(3,3,1);
	
	set_pixel(im, 0, 0, 0, -1);
	set_pixel(im, 1, 0, 0, -2);
	set_pixel(im, 2, 0, 0, -1);
	set_pixel(im, 0, 1, 0, 0);
	set_pixel(im, 1, 1, 0, 0);
	set_pixel(im, 2, 1, 0, 0);
	set_pixel(im, 0, 2, 0, 1);
	set_pixel(im, 1, 2, 0, 2);
	set_pixel(im, 2, 2, 0, 1);
    return im;
}

//Scale the image so all values lie between [0-1]. 
void feature_normalize(image im)
{
	int i,j,k;
	float min, max;
	for(k=0; k<im.c; ++k){
		min = 0;
		max = 0; 
		for(j=0; j<im.h; ++j){
			for(i=0; i<im.w; ++i){
				float v = get_pixel(im, i, j, k);
				// find the min and max value in the image.
				min = ( v-min > 0) ? min : v;
				max = ( v-max > 0) ? v : max;
			}
		}
		for(j=0; j<im.h; ++j){
			for(i=0; i<im.w; ++i){
				// if range(max-min) is zero, set value to zero
				// if not, normalized_value = v - min / max-min
				float v = ( max-min == 0 ) ? 0 : (get_pixel(im, i, j, k)-min) / (max-min); 
				set_pixel(im, i, j, k, v);
			}
		}
	}
}


// return two images, the gradient magnitude and direction.
image *sobel_image(image im)
{
	image gx = make_gx_filter();
	image gy = make_gy_filter();

   	image im_gx = convolve_image(im, gx, 0);
    image im_gy= convolve_image(im, gy, 0);
	
	image *res = calloc(2, sizeof(im.w*im.h));
    res[0] = make_image(im.w, im.h, 1);
    res[1] = make_image(im.w, im.h, 1);

	int i,j; 
	for(j=0; j<im.h; ++j){
		for(i=0; i<im.w; ++i){
			float x = get_pixel(im_gx, i, j, 0); 
			float y = get_pixel(im_gy, i, j, 0);
			float mag = sqrtf(x*x + y*y); // first image is for magnitude
			float dir = atan2f(y, x);	// second image is for direction
			set_pixel(res[0], i, j, 0, mag); 
			set_pixel(res[1], i, j, 0, dir); 
		}
	}	
	
	return res;
}

// coloarize image 
image colorize_sobel(image im)
{
	//magnitude to specify the saturation and value of an image and the angle to specify the hue 
    image im_c = make_image(im.w, im.h, 3);
	rgb_to_hsv(im_c);	
	
	image f = make_gaussian_filter(1); // First, apply guassian filter for smoothing
	image blur = convolve_image(im, f, 1);
	image* res = sobel_image(blur); // sobel fiter to get manitude and direction values 
	printf("%d %d %d\n", im_c.w, im_c.h, im_c.c);
	printf("%d %d %d\n", res[0].w, res[0].h, res[0].c);

	int i,j; 
	for(j=0; j<im_c.h; ++j){
		for(i=0; i<im_c.w; ++i){
			float v = get_pixel(res[1], i, j, 0);   //Used angle(direction) to specify the hue	
			set_pixel(im_c, i, j, 0, v*0.2); 
			v = get_pixel(res[0], i, j, 0);   //Used magnitude to specify the stauration and value
			set_pixel(im_c, i, j, 1, v*0.2); 
			set_pixel(im_c, i, j, 2, v*0.3); 
		}
	}

	clamp_image(im_c);
	hsv_to_rgb(im_c);

    return im_c;
}
