#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#include "matrix.h"

// Draws a line on an image with color corresponding to the direction of line
// image im: image to draw line on
// float x, y: starting point of line
// float dx, dy: vector corresponding to line angle and magnitude
void draw_line(image im, float x, float y, float dx, float dy)
{
    assert(im.c == 3);
    float angle = 6*(atan2(dy, dx) / TWOPI + .5);
    int index = floor(angle);
    float f = angle - index;
    float r, g, b;
    if(index == 0){
        r = 1; g = f; b = 0;
    } else if(index == 1){
        r = 1-f; g = 1; b = 0;
    } else if(index == 2){
        r = 0; g = 1; b = f;
    } else if(index == 3){
        r = 0; g = 1-f; b = 1;
    } else if(index == 4){
        r = f; g = 0; b = 1;
    } else {
        r = 1; g = 0; b = 1-f;
    }
    float i;
    float d = sqrt(dx*dx + dy*dy);
    for(i = 0; i < d; i += 1){
        int xi = x + dx*i/d;
        int yi = y + dy*i/d;
        set_pixel(im, xi, yi, 0, r);
        set_pixel(im, xi, yi, 1, g);
        set_pixel(im, xi, yi, 2, b);
    }
}

// Make an integral image or summed area table from an image
// image im: image to process
// returns: image I such that I[x,y] = sum{i<=x, j<=y}(im[i,j])
image make_integral_image(image im)
{
    image integ = make_image(im.w, im.h, im.c);
    // TODO: fill in the integral image
	// Summed-area table
	int i, j, k;
	for(k=0; k<im.c; ++k){
		for(j=0; j<im.h; ++j){
			for(i=0; i<im.w; ++i){
				int pixel_index = i + im.w*j + im.w*im.h*k;
				if(i==0 && j==0){ //[0][0] image copy
					integ.data[pixel_index] = im.data[pixel_index];		
				}else if(i==0){ //first column
					integ.data[pixel_index] = im.data[pixel_index] + integ.data[pixel_index-im.w] ;
				}else if(j==0){ //first row
					integ.data[pixel_index] = im.data[pixel_index] + integ.data[pixel_index-1] ;
				}else{ //the rest I(x,y)=i(x,y)+I(x,y-1)+I(x-1,y)-I(x-1,y-1)
					integ.data[pixel_index] = im.data[pixel_index] + integ.data[pixel_index-1]+integ.data[pixel_index-im.w]-integ.data[pixel_index-im.w-1];
				}
			}
		}
	}

    return integ;
}

int get_pixel_bound_x(image im, int x)
{
    // Padding. ex) Indexing '-20' retrun 0. Indexing '300' for 255 image, return 255-1
    x = (x<-1) ? -1 : x;
    x = (x>=im.w) ? (im.w-1) : x;
    return x;
}               

int get_pixel_bound_y(image im, int y)
{
    // Padding. ex) Indexing '-20' retrun 0. Indexing '300' for 255 image, return 255-1
    y = (y<-1) ? -1 : y;
    y = (y>=im.h) ? (im.h-1) : y;
    return y;
}               

// Apply a box filter to an image using an integral image for speed
// image im: image to smooth
// int s: window size for box filter
// returns: smoothed image
image box_filter_image(image im, int s)
{
    int i,j,k;
    image integ = make_integral_image(im);
    image S = make_image(im.w, im.h, im.c);
	for(k=0; k<im.c; ++k){
		for(j=0; j<im.h+s/2; ++j){
			for(i=0; i<im.w+s/2; ++i){
				float A_v = get_pixel(integ, i-s/2-1, j-s/2-1, k);
				int A_v_x = get_pixel_bound_x(integ, i-s/2-1);
				int A_v_y = get_pixel_bound_y(integ, j-s/2-1);
				if(A_v_x==-1 || A_v_y==-1){
					A_v = 0; 
				}
				float B_v = get_pixel(integ, i+s/2, j-s/2-1, k);
				int B_v_x = get_pixel_bound_x(integ, i+s/2);
				int B_v_y = get_pixel_bound_y(integ, j-s/2-1);
				if(B_v_y==-1){
					B_v = 0; 
				}
				float C_v = get_pixel(integ, i-s/2-1, j+s/2, k);
				int C_v_x = get_pixel_bound_y(integ, i-s/2-1);
				int C_v_y = get_pixel_bound_y(integ, j+s/2);
				if(C_v_x==-1){
					C_v = 0; 
				}
				float D_v = get_pixel(integ, i+s/2, j+s/2, k);
				float v = D_v + A_v - B_v - C_v; // D+A-B-C
				int norm = (B_v_x - A_v_x) * (C_v_y - A_v_y); //Normalize by the size of window
				set_pixel(S, i, j, k, v/(float)norm);
			}
		}
	}
    return S;
}

// Calculate the time-structure matrix of an image pair.
// image im: the input image.
// image prev: the previous image in sequence.
// int s: window size for smoothing.
// returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2,
//          3rd channel is IxIy, 4th channel is IxIt, 5th channel is IyIt.
image time_structure_matrix(image im, image prev, int s)
{
    int converted = 0;
    if(im.c == 3){
        converted = 1;
        im = rgb_to_grayscale(im);
        prev = rgb_to_grayscale(prev);
    }

    // TODO: calculate gradients, structure components, and smooth them
    image S = make_image(im.w, im.h, 5);

    // 1.Calculate image derivatives Ix and Iy. --> Sobel filter
    image gx = make_gx_filter();
    image gy = make_gy_filter();
    image Ix = convolve_image(im, gx, 0);
    image Iy = convolve_image(im, gy, 0);

    // 2.Calculate measures IxIx, IyIy, and IxIy.
    int i,j;
    for(j=0; j<im.h; ++j){
        for(i=0; i<im.w; ++i){
            float ix = get_pixel(Ix, i, j, 0);
            float iy = get_pixel(Iy, i, j, 0);
			float it = get_pixel(im, i, j, 0)-get_pixel(prev, i, j, 0); //it =im - prev
			//float it = get_pixel(prev, i, j, 0)-get_pixel(im, i, j, 0); //it = prev - im
            set_pixel(S, i, j, 0, ix*ix); //1st channel is Ix^2
            set_pixel(S, i, j, 1, iy*iy); //2nd channel is Iy^2
            set_pixel(S, i, j, 2, ix*iy); //3rd channel is Ix*Iy
			set_pixel(S, i, j, 3, ix*it); //4th channel is Ix*It
			set_pixel(S, i, j, 4, iy*it); //5th channel is Iy*It
        }
    }
	//smooth
	S = box_filter_image(S,s);

    if(converted){
        free_image(im); free_image(prev);
    }
    return S;
}

// Calculate the velocity given a structure image
// image S: time-structure image
// int stride: only calculate subset of pixels for speed
image velocity_image(image S, int stride)
{
    image v = make_image(S.w/stride, S.h/stride, 3);
    int i, j;
    matrix M = make_matrix(2,2);
    for(j = (stride-1)/2; j < S.h; j += stride){
        for(i = (stride-1)/2; i < S.w; i += stride){
            float Ixx = S.data[i + S.w*j + 0*S.w*S.h];
            float Iyy = S.data[i + S.w*j + 1*S.w*S.h];
            float Ixy = S.data[i + S.w*j + 2*S.w*S.h];
            float Ixt = S.data[i + S.w*j + 3*S.w*S.h];
            float Iyt = S.data[i + S.w*j + 4*S.w*S.h];

            // TODO: calculate vx and vy using the flow equation
            float vx = 0;
            float vy = 0;
			M.data[0][0] = Ixx; 
			M.data[0][1] = Ixy;
			M.data[1][0] = Ixy;
			M.data[1][1] = Iyy;
			matrix Minv = matrix_invert(M); 
			if(Minv.rows == 0 || Minv.cols == 0){ // set the velocity to zero if the matrix is not invertible
			//if(Minv.rows != 2 || Minv.cols != 2){ // set the velocity to zero if the matrix is not invertible
				vx = 0;
				vy = 0; 
			}else{
				vx = -1*(Minv.data[0][0]*Ixt + Minv.data[0][1]*Iyt); 
				vy = -1*(Minv.data[1][0]*Ixt + Minv.data[1][1]*Iyt); 
			}
            set_pixel(v, i/stride, j/stride, 0, vx);
            set_pixel(v, i/stride, j/stride, 1, vy);
        }
    }
    free_matrix(M);
    return v;
}

// Draw lines on an image given the velocity
// image im: image to draw on
// image v: velocity of each pixel
// float scale: scalar to multiply velocity by for drawing
void draw_flow(image im, image v, float scale)
{
    int stride = im.w / v.w;
    int i,j;
    for (j = (stride-1)/2; j < im.h; j += stride) {
        for (i = (stride-1)/2; i < im.w; i += stride) {
            float dx = scale*get_pixel(v, i/stride, j/stride, 0);
            float dy = scale*get_pixel(v, i/stride, j/stride, 1);
            if(fabs(dx) > im.w) dx = 0;
            if(fabs(dy) > im.h) dy = 0;
            draw_line(im, i, j, dx, dy);
        }
    }
}


// Constrain the absolute value of each image pixel
// image im: image to constrain
// float v: each pixel will be in range [-v, v]
void constrain_image(image im, float v)
{
    int i;
    for(i = 0; i < im.w*im.h*im.c; ++i){
        if (im.data[i] < -v) im.data[i] = -v;
        if (im.data[i] >  v) im.data[i] =  v;
    }
}

// Calculate the optical flow between two images
// image im: current image
// image prev: previous image
// int smooth: amount to smooth structure matrix by
// int stride: downsampling for velocity matrix
// returns: velocity matrix
image optical_flow_images(image im, image prev, int smooth, int stride)
{
    image S = time_structure_matrix(im, prev, smooth);   
    image v = velocity_image(S, stride);
    constrain_image(v, 6);
    image vs = smooth_image(v, 2);
    free_image(v);
    free_image(S);
    return vs;
}

// Run optical flow demo on webcam
// int smooth: amount to smooth structure matrix by
// int stride: downsampling for velocity matrix
// int div: downsampling factor for images from webcam
void optical_flow_webcam(int smooth, int stride, int div)
{
#ifdef OPENCV
    void * cap;
    cap = open_video_stream(0, 0, 1280, 720, 30);
    image prev = get_image_from_stream(cap);
    image prev_c = nn_resize(prev, prev.w/div, prev.h/div);
    image im = get_image_from_stream(cap);
    image im_c = nn_resize(im, im.w/div, im.h/div);
    while(im.data){
        image copy = copy_image(im);
        image v = optical_flow_images(im_c, prev_c, smooth, stride);
        draw_flow(copy, v, smooth*div);
        int key = show_image(copy, "flow", 5);
        free_image(v);
        free_image(copy);
        free_image(prev);
        free_image(prev_c);
        prev = im;
        prev_c = im_c;
        if(key != -1) {
            key = key % 256;
            printf("%d\n", key);
            if (key == 27) break;
        }
        im = get_image_from_stream(cap);
        im_c = nn_resize(im, im.w/div, im.h/div);
    }
#else
    fprintf(stderr, "Must compile with OpenCV\n");
#endif
}
