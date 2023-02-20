#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include "image.h"
#include "matrix.h"
#define FLT_MAX 3.4028234664e+38
#define NEG_INT_MAX (-1*2147483648)

// Comparator for matches
// const void *a, *b: pointers to the matches to compare.
// returns: result of comparison, 0 if same, 1 if a > b, -1 if a < b.
int match_compare(const void *a, const void *b)
{
    match *ra = (match *)a;
    match *rb = (match *)b;
    if (ra->distance < rb->distance) return -1;
    else if (ra->distance > rb->distance) return  1;
    else return 0;
}

// Helper function to create 2d points.
// float x, y: coordinates of point.
// returns: the point.
point make_point(float x, float y)
{
    point p;
    p.x = x; p.y = y;
    return p;
}

// Place two images side by side on canvas, for drawing matching pixels.
// image a, b: images to place.
// returns: image with both a and b side-by-side.
image both_images(image a, image b)
{
    image both = make_image(a.w + b.w, a.h > b.h ? a.h : b.h, a.c > b.c ? a.c : b.c);
    int i,j,k;
    for(k = 0; k < a.c; ++k){
        for(j = 0; j < a.h; ++j){
            for(i = 0; i < a.w; ++i){
                set_pixel(both, i, j, k, get_pixel(a, i, j, k));
            }
        }
    }
    for(k = 0; k < b.c; ++k){
        for(j = 0; j < b.h; ++j){
            for(i = 0; i < b.w; ++i){
                set_pixel(both, i+a.w, j, k, get_pixel(b, i, j, k));
            }
        }
    }
    return both;
}

// Draws lines between matching pixels in two images.
// image a, b: two images that have matches.
// match *matches: array of matches between a and b.
// int n: number of matches.
// int inliers: number of inliers at beginning of matches, drawn in green.
// returns: image with matches drawn between a and b on same canvas.
image draw_matches(image a, image b, match *matches, int n, int inliers)
{
    image both = both_images(a, b);
    int i,j;
    for(i = 0; i < n; ++i){
        int bx = matches[i].p.x; 
        int ex = matches[i].q.x; 
        int by = matches[i].p.y;
        int ey = matches[i].q.y;
        for(j = bx; j < ex + a.w; ++j){
            int r = (float)(j-bx)/(ex+a.w - bx)*(ey - by) + by;
            set_pixel(both, j, r, 0, i<inliers?0:1);
            set_pixel(both, j, r, 1, i<inliers?1:0);
            set_pixel(both, j, r, 2, 0);
        }
    }
    return both;
}

// Draw the matches with inliers in green between two images.
// image a, b: two images to match.
// matches *
image draw_inliers(image a, image b, matrix H, match *m, int n, float thresh)
{
    int inliers = model_inliers(H, m, n, thresh);
    image lines = draw_matches(a, b, m, n, inliers);
    return lines;
}

// Find corners, match them, and draw them between two images.
// image a, b: images to match.
// float sigma: gaussian for harris corner detector. Typical: 2
// float thresh: threshold for corner/no corner. Typical: 1-5
// int nms: window to perform nms on. Typical: 3
image find_and_draw_matches(image a, image b, float sigma, float thresh, int nms)
{
    int an = 0;
    int bn = 0;
    int mn = 0;
    descriptor *ad = harris_corner_detector(a, sigma, thresh, nms, &an);
    descriptor *bd = harris_corner_detector(b, sigma, thresh, nms, &bn);
    match *m = match_descriptors(ad, an, bd, bn, &mn);

    mark_corners(a, ad, an);
    mark_corners(b, bd, bn);
    image lines = draw_matches(a, b, m, mn, 0);

    free_descriptors(ad, an);
    free_descriptors(bd, bn);
    free(m);
    return lines;
}

// Calculates L1 distance between to floating point arrays.
// float *a, *b: arrays to compare.
// int n: number of values in each array.
// returns: l1 distance between arrays (sum of absolute differences).
float l1_distance(float *a, float *b, int n)
{
    // TODO: return the correct number.
	int i;
	float sum = 0;
	for(i=0; i<n; ++i){
		sum += fabs(a[i] - b[i]);
	}
    return sum;
}

// Finds best matches between descriptors of two images.
// descriptor *a, *b: array of descriptors for pixels in two images.
// int an, bn: number of descriptors in arrays a and b.
// int *mn: pointer to number of matches found, to be filled in by function.
// returns: best matches found. each descriptor in a should match with at most
//          one other descriptor in b.
match *match_descriptors(descriptor *a, int an, descriptor *b, int bn, int *mn)
{
    int i,j,k;

    // We will have at most an matches.
    *mn = an;
    match *m = calloc(an, sizeof(match));
    for(j = 0; j < an; ++j){
        // TODO: for every descriptor in a, find best match in b.
		float smallest_l1_dist = FLT_MAX;
        int bind = 0; // <- find the best match
		for(i=0; i<bn; ++i){
			float l1_dist = l1_distance(a[j].data, b[i].data, (a->n));
			if(l1_dist < smallest_l1_dist){
				smallest_l1_dist = l1_dist;
				bind = i;
			}
		}

        // record ai as the index in *a and bi as the index in *b.
        m[j].ai = j;
        m[j].bi = bind; // <- should be index in b.
        m[j].p = a[j].p;
        m[j].q = b[bind].p;
        m[j].distance = smallest_l1_dist; // <- should be the smallest L1 distance!
    }

    int count = 0;
	int offset =0;
    int *seen = calloc(bn, sizeof(int));
    // TODO: we want matches to be injective (one-to-one).
    // Sort matches based on distance using match_compare and qsort.
	//int com = match_compare(m[], m[]); //result of comparison, 0 if same, 1 if a > b, -1 if a < b.
	qsort(m, an, sizeof(match), match_compare);
	// Then throw out matches to the same element in b. Use seen to keep track.
    for(i=0; i<an; ++i){
        int seen_i = m[i].bi;
        if( seen[seen_i] != 0){
            for(j=i; j<an; j++){ //shift array
                k=j+1;
                *(m+j)=*(m+k);
            }
            offset += 1;
            i-=1;
            an-=1;
		}else{
			count += 1;
		}
		seen[seen_i] += 1;  
	}
    // Each point should only be a part of one match.
    // Some points will not be in a match.
    // In practice just bring good matches to front of list, set *mn
    *mn = count;
    free(seen);
    return m;
}

// Apply a projective transformation to a point.
// matrix H: homography to project point.
// point p: point to project.
// returns: point projected using the homography.
point project_point(matrix H, point p)
{
    matrix c = make_matrix(3, 1);
    // TODO: project point p with homography H.
    // Remember that homogeneous coordinates are equivalent up to scalar.
    // Have to divide by.... something...
	c.data[0][0] = p.x; 
	c.data[1][0] = p.y;
	c.data[2][0] = 1;

	matrix x_prime = matrix_mult_matrix(H, c);
	float x = x_prime.data[0][0]/x_prime.data[2][0];
	float y = x_prime.data[1][0]/x_prime.data[2][0];

	point q = make_point(x, y);
    return q;
}

// Calculate L2 distance between two points.
// point p, q: points.
// returns: L2 distance between them.
float point_distance(point p, point q)
{
    // TODO: should be a quick one.
	//return sqrtf(pow((p.x-q.x),2) + pow((p.y-q.y),2));
	float v = (p.x-q.x)*(p.x-q.x) + (p.y-q.y)*(p.y-q.y); 
	if( v >= FLT_MAX){
		v = FLT_MAX;
	}
	else{
		v = sqrtf(v);
	}
	//return sqrtf((p.x-q.x)*(p.x-q.x) + (p.y-q.y)*(p.y-q.y));
	//printf("v: %f\n",v);
	return v; 
}

// Count number of inliers in a set of matches. Should also bring inliers
// to the front of the array.
// matrix H: homography between coordinate systems.
// match *m: matches to compute inlier/outlier.
// int n: number of matches in m.
// float thresh: threshold to be an inlier.
// returns: number of inliers whose projected point falls within thresh of
//          their match in the other image. Should also rearrange matches
//          so that the inliers are first in the array. For drawing.
int model_inliers(matrix H, match *m, int n, float thresh)
{
    int i;
    int count = 0;
    // TODO: count number of matches that are inliers
	// Loop over the points, project using the homography
	if(H.rows == 0 || H.cols ==0){
		printf("H.rows(%d), H.cols(%d)\n ",H.rows, H.cols);
		return 0;	
	}
	for(i=0; i<n; ++i){
		point p = project_point(H, m[i].p);
		point q = m[i].q;
	 	float dist = point_distance(p,q); // i.e. distance(H*p, q) < thresh
		//count += (dist < thresh) ? 1 : 0;
		if( dist < thresh){
    	// Also, sort the matches m so the inliers are the first 'count' elements.
			match tmp = *(m+i);
			*(m+i) = *(m+count);
			*(m+count) = tmp;
			count  += 1;
		}
	}

    return count;
}

// Randomly shuffle matches for RANSAC.
// match *m: matches to shuffle in place.
// int n: number of elements in matches.
void randomize_matches(match *m, int n)
{
    // TODO: implement Fisher-Yates to shuffle the array.
	//for i from n−1 downto 1 do
	//     j ← random integer such that 0 ≤ j ≤ i
	//     exchange a[j] and a[i]
	int i,j;
	for(i=n-1; i>0; --i){
		j = rand() % i;
		match tmp = *(m+j);
		*(m+j) = *(m+i);
		*(m+i) = tmp;
	}
}

// Computes homography between two images given matching pixels.
// match *matches: matching points between images.
// int n: number of matches to use in calculating homography.
// returns: matrix representing homography H that maps image a to image b.
matrix compute_homography(match *matches, int n)
{
    matrix M = make_matrix(n*2, 8);
    matrix b = make_matrix(n*2, 1);

    int i;
    for(i = 0; i < n; ++i){
        double x  = matches[i].p.x;
        double xp = matches[i].q.x;
        double y  = matches[i].p.y;
        double yp = matches[i].q.y;
        // TODO: fill in the matrices M and b.
		//| x, y, 1, 0, 0, 0, -Mx*Nx, -My*Nx | <--- Matrix M
		//| 0, 0, 0, x, y, 1, -Mx*Ny, -Ny*Ny | 
		M.data[2*i][0] = x;
		M.data[2*i][1] = y; 
		M.data[2*i][2] = 1;
		M.data[2*i][3] = 0;
		M.data[2*i][4] = 0;
		M.data[2*i][5] = 0;
		M.data[2*i][6] = -1*x*xp;
		M.data[2*i][7] = -1*y*xp;
		M.data[2*i+1][0] = 0;
		M.data[2*i+1][1] = 0; 
		M.data[2*i+1][2] = 0;
		M.data[2*i+1][3] = x;
		M.data[2*i+1][4] = y;
		M.data[2*i+1][5] = 1;
		M.data[2*i+1][6] = -1*x*yp;
		M.data[2*i+1][7] = -1*y*yp;
		//Matrix b
		b.data[2*i][0] = xp;
		b.data[2*i+1][0] = yp;
    }
    matrix a = solve_system(M, b);
    free_matrix(M); free_matrix(b); 

    // If a solution can't be found, return empty matrix;
    matrix none = {0};
    if(!a.data){ printf("Cannot find the solution of H\n");}
    if(!a.data) return none;

    matrix H = make_matrix(3, 3);
    // TODO: fill in the homography H based on the result in a.
	H.data[0][0] = a.data[0][0];
	H.data[0][1] = a.data[1][0];
	H.data[0][2] = a.data[2][0];
	H.data[1][0] = a.data[3][0];
	H.data[1][1] = a.data[4][0];
	H.data[1][2] = a.data[5][0];
	H.data[2][0] = a.data[6][0];
	H.data[2][1] = a.data[7][0];
	H.data[2][2] = 1;
	
    free_matrix(a);
    return H;
}

// Perform RANdom SAmple Consensus to calculate homography for noisy matches.
// match *m: set of matches.
// int n: number of matches.
// float thresh: inlier/outlier distance threshold.
// int k: number of iterations to run.
// int cutoff: inlier cutoff to exit early.
// returns: matrix representing most common homography between matches.
matrix RANSAC(match *m, int n, float thresh, int k, int cutoff)
{
    int e=0;
	int best = 0;  //bestfit = INF  //greater than 0 for homography     computation

    matrix Hb = make_translation_homography(256, 0);
    // TODO: fill in RANSAC algorithm.
    // for k iterations:
    //     shuffle the matches
    //     compute a homography with a few matches (how many??)
    //     if new homography is better than old (how can you tell?):
    //         compute updated homography using all inliers
    //         remember it and how good it is
    //         if it's better than the cutoff:
    //             return it immediately
    // if we get to the end return the best homography

	while(e<k){ // for k iterations:
		e += 1; 
		randomize_matches(m, n); //shuffle the matches
		matrix H = compute_homography(m, 4); //compute a homography with a few matches (how many??-->4)
		int inliers = model_inliers(H, m, n, thresh); //inliers = data within t of model 
		if(inliers > best){ //if new homography is better than old (using inliers)  
			Hb = compute_homography(m, inliers); //compute updated homography using all inliers
			best = inliers;
		}
		if(inliers > cutoff){
			//printf("best > cutoff: %d\n",best);
			return Hb;
		}
	}
	//printf("inliner: %d from matches %d\n", best, n);
	return Hb;
}

// Stitches two images together using a projective transformation.
// image a, b: images to stitch.
// matrix H: homography from image a coordinates to image b coordinates.
// returns: combined image stitched together.
image combine_images(image a, image b, matrix H)
{
    matrix Hinv = matrix_invert(H);
    
	save_image(b, "b_test");

    // Project the corners of image b into image a coordinates.
    point c1 = project_point(Hinv, make_point(0,0)); //topleft
    point c2 = project_point(Hinv, make_point(b.w-1, 0)); //topright
    point c3 = project_point(Hinv, make_point(0, b.h-1)); //botleft
    point c4 = project_point(Hinv, make_point(b.w-1, b.h-1)); //botright
/*
	printf("c1:(%f,%f)\n",c1.x, c1.y);
	printf("c2:(%f,%f)\n",c2.x, c2.y);
	printf("c3:(%f,%f)\n",c3.x, c3.y);
	printf("c4:(%f,%f)\n",c4.x, c4.y);

    point d1 = project_point(H, make_point(c1.x, c1.y)); //topleft of b
	//printf("org:(%f,%f)\n\n",d1.x, d1.y);
*/	
	set_pixel(a, floor(c1.x), floor(c1.y), 0, 255);
	set_pixel(a, floor(c1.x), floor(c1.y), 1, 0);
	set_pixel(a, floor(c1.x), floor(c1.y), 2, 0);
	save_image(a, "a_test");
	

    // Find top left and bottom right corners of image b warped into image a.
    point topleft, botright;
    topleft.x = MIN(c1.x, MIN(c2.x, MIN(c3.x, c4.x)));
    topleft.y = MIN(c1.y, MIN(c2.y, MIN(c3.y, c4.y)));
    botright.x = MAX(c1.x, MAX(c2.x, MAX(c3.x, c4.x)));
    botright.y = MAX(c1.y, MAX(c2.y, MAX(c3.y, c4.y)));
	//printf("topleft:(%f,%f)\n",topleft.x, topleft.y);
	//printf("botright:(%f,%f)\n\n",botright.x, botright.y);
    
	// Find how big our new image should be and the offsets from image a.
    int dx = MIN(0, topleft.x);
    int dy = MIN(0, topleft.y);
    int w = MAX(a.w, botright.x) - dx;
    int h = MAX(a.h, botright.y) - dy;
	//printf("dx dy:(%d,%d), w y:(%d,%d)\n\n",dx,dy,w,h);

    // Can disable this if you are making very big panoramas.
    // Usually this means there was an error in calculating H.
	/*
    if(w > 7000 || h > 7000){
        fprintf(stderr, "output too big, stopping\n");
        return copy_image(a);
    }
	*/

    int i,j,k;
    image c = make_image(w, h, a.c);
	//printf("c image size) c.w(%d), c.h(%d) \n", w, h);
    for(k = 0; k < a.c; ++k){
        for(j = 0; j < c.h; ++j){
            for(i = 0; i < c.w; ++i){
				set_pixel(c, i, j, k, 150);
            }
        }
    }
    
    // Paste image a into the new image offset by dx and dy.
    for(k = 0; k < a.c; ++k){
        for(j = 0; j < a.h; ++j){
            for(i = 0; i < a.w; ++i){
                // TODO: fill in.
				float v = get_pixel(a, i, j, k);
				set_pixel(c, i-dx, j-dy, k, v);
            }
        }
    }

    // TODO: Paste in image b as well.
    // You should loop over some points in the new image (which? all?)
    // and see if their projection from a coordinates to b coordinates falls
    // inside of the bounds of image b. If so, use bilinear interpolation to
    // estimate the value of b at that projection, then fill in image c.
    for(k = 0; k < a.c; ++k){ //loop over points in a 
        for(j = topleft.y; j < botright.y; ++j){ //loop over: topleft y ~ botright y
            for(i = topleft.x ; i < botright.x; ++i){ //loop over: topleft x ~ botright x
    			point b1 = project_point(H, make_point(i, j)); //topleft of b
				if( b1.x >= 0 && b1.x <= b.w-1 && b1.y >= 0 && b1.y <= b.h-1){
					float v = bilinear_interpolate(b, b1.x, b1.y, k);
					set_pixel(c, i-dx, j-dy, k, v);
				}
            }
        }
    }

    return c;
}

// Create a panoramam between two images.
// image a, b: images to stitch together.
// float sigma: gaussian for harris corner detector. Typical: 2
// float thresh: threshold for corner/no corner. Typical: 1-5
// int nms: window to perform nms on. Typical: 3
// float inlier_thresh: threshold for RANSAC inliers. Typical: 2-5
// int iters: number of RANSAC iterations. Typical: 1,000-50,000
// int cutoff: RANSAC inlier cutoff. Typical: 10-100
image panorama_image(image a, image b, float sigma, float thresh, int nms, float inlier_thresh, int iters, int cutoff)
{
    srand(10);
    int an = 0;
    int bn = 0;
    int mn = 0;
    
    // Calculate corners and descriptors
    descriptor *ad = harris_corner_detector(a, sigma, thresh, nms, &an);
    descriptor *bd = harris_corner_detector(b, sigma, thresh, nms, &bn);

    // Find matches
    match *m = match_descriptors(ad, an, bd, bn, &mn);

    // Run RANSAC to find the homography
    matrix H = RANSAC(m, mn, inlier_thresh, iters, cutoff);

    if(0){
        // Mark corners and matches between images
		print_matrix(H);
        mark_corners(a, ad, an);
        mark_corners(b, bd, bn);
        image inlier_matches = draw_inliers(a, b, H, m, mn, inlier_thresh);
        save_image(inlier_matches, "inliers");
    }

    free_descriptors(ad, an);
    free_descriptors(bd, bn);
    free(m);

    // Stitch the images together with the homography
    image comb = combine_images(a, b, H);
    return comb;
}

// Project an image onto a cylinder.
// image im: image to project.
// float f: focal length used to take image (in pixels).
// returns: image projected onto cylinder, then flattened.
image cylindrical_project(image im, float f)
{
    //TODO: project image onto a cylinder
    image c = copy_image(im);
    return c;
}
