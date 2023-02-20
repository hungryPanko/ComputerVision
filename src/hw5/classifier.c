#include <math.h>
#include <stdlib.h>
#include "image.h"
#include "matrix.h"

// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        double sum = 0;
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            if(a == LOGISTIC){
				//m.data[i][j] = expf(x) / (1 + expf(x));
				m.data[i][j] = exp(x) / (1 + exp(x)); // e^x / (1 + e^x) 
            } else if (a == RELU){
				m.data[i][j] = (x>0) ? x : 0;     // y = max(0, x)
            } else if (a == LRELU){
				float max = (x>0) ? x : 0;  // f(x)=max(0,x)+β∗min(0,x)
				float min = (x>0) ? 0: x;
				min = 0.1*min; 
				m.data[i][j] = max + min; 
            } else if (a == SOFTMAX){
				m.data[i][j] = exp(x);  
            }
            sum += m.data[i][j];
        }
        if (a == SOFTMAX) {
            // TODO: have to normalize by sum if we are using SOFTMAX
        	for(j = 0; j < m.cols; ++j){
            	m.data[i][j] = m.data[i][j] / sum;
			}
			sum = 0;
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
			double y = d.data[i][j];
            // TODO: multiply the correct element of d by the gradient
            if(a == LOGISTIC){
				d.data[i][j] = x*(1-x) * y; //f'(x) = f(x) * (1 - f(x))
            } else if (a == RELU){
				d.data[i][j] = (x>0) ? y : 0; //f'(x) = 0 (x<0), f'(x) = gradient 
            } else if (a == LRELU){
				d.data[i][j] = (x>0) ? y : 0.1*y; //f'(x) = 0 (x<0), f'(x) = gradient 
            } else if (a == SOFTMAX){
				d.data[i][j] = y; 
            }
        }
    }
}

// Forward propagate information through a layer
// layer *l: pointer to the layer
// matrix in: input to layer
// returns: matrix that is output of the layer
matrix forward_layer(layer *l, matrix in)
{

    l->in = in;  // Save the input for backpropagation

    // TODO: fix this! multiply input by weights and apply activation function.
	matrix out = matrix_mult_matrix(l->in, l->w); 	// X * Weight
	activate_matrix(out, l->activation); 			//F( X * Weight)

    free_matrix(l->out);// free the old output
    l->out = out;       // Save the current output for gradient calculation
    return out;
}

// Backward propagate derivatives through a layer
// layer *l: pointer to the layer
// matrix delta: partial derivative of loss w.r.t. output of layer
// returns: matrix, partial derivative of loss w.r.t. input to layer
matrix backward_layer(layer *l, matrix delta)
{
    // 1.4.1
    // delta is dL/dy
    // TODO: modify it in place to be dL/d(xw)
	//dL/d(xw) = dL/dy * dy/d(xw)
	//         = dL/dy * df(xw)/d(xw)
	//         = dL/dy * f'(xw)
	gradient_matrix(l->out, l->activation, delta);    //from dL/dy to dL/d(xw)

    // 1.4.2
    // TODO: then calculate dL/dw and save it in l->dw
    free_matrix(l->dw);
    //matrix dw = make_matrix(l->w.rows, l->w.cols); // replace this
	//	dL/dw = dL/d(xw) * d(xw)/dw
    //  	  = dL/d(xw) * x   (xt * dL/d(xw))
	//delta is dL/d(xw) 
	l->dw = matrix_mult_matrix(transpose_matrix(l->in), delta);
    
    // 1.4.3
    // TODO: finally, calculate dL/dx and return it.
    // matrix dx = make_matrix(l->in.rows, l->in.cols); // replace this
	// dL/dx = dL/d(xw) * d(xw)/dx
    //       = dL/d(xw) * w   (dL/d(xw) * wt)
	matrix dx = matrix_mult_matrix(delta, transpose_matrix(l->w));

    return dx;
}

// Update the weights at layer l
// layer *l: pointer to the layer
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_layer(layer *l, double rate, double momentum, double decay)
{
    // TODO:
    // Calculate Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
	// l->v  = l->dw  -λ(weight decay)*l->w +  m*l->v
	// weighted sum of l->dw, l->w, and l->v
    // save it to l->v
	matrix tmp1 = axpy_matrix(-1*decay, l->w, l->dw);
	//l->v = axpy_matrix(momentum, l->v, axpy_matrix(-1*decay, l->w, l->dw)); //ok
	matrix tmp2 = axpy_matrix(momentum, l->v, tmp1); //ok
	free_matrix(l->v);
	l->v = copy_matrix(tmp2);

	free_matrix(tmp1);
	free_matrix(tmp2);
	
    // Update l->w
	//w_{t+1} = w_t + ηΔw_t
	//l->w = axpy_matrix(rate, l->v, l->w); //ok
	matrix tmp3 = axpy_matrix(rate, l->v, l->w); 
	free_matrix(l->w);
	l->w = copy_matrix(tmp3);
	free_matrix(tmp3);

    // Remember to free any intermediate results to avoid memory leaks
}

// Make a new layer for our model
// int input: number of inputs to the layer
// int output: number of outputs from the layer
// ACTIVATION activation: the activation function to use
layer make_layer(int input, int output, ACTIVATION activation)
{
    layer l;
    l.in  = make_matrix(1,1);
    l.out = make_matrix(1,1);
    l.w   = random_matrix(input, output, sqrt(2./input));
    l.v   = make_matrix(input, output);
    l.dw  = make_matrix(input, output);
    l.activation = activation;
    return l;
}

// Run a model on input X
// model m: model to run
// matrix X: input to model
// returns: result matrix
matrix forward_model(model m, matrix X)
{
    int i;
    for(i = 0; i < m.n; ++i){
        X = forward_layer(m.layers + i, X);
    }
    return X;
}

// Run a model backward given gradient dL
// model m: model to run
// matrix dL: partial derivative of loss w.r.t. model output dL/dy
void backward_model(model m, matrix dL)
{
    matrix d = copy_matrix(dL);
    int i;
    for(i = m.n-1; i >= 0; --i){
        matrix prev = backward_layer(m.layers + i, d);
        free_matrix(d);
        d = prev;
    }
    free_matrix(d);
}

// Update the model weights
// model m: model to update
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_model(model m, double rate, double momentum, double decay)
{
    int i;
    for(i = 0; i < m.n; ++i){
        update_layer(m.layers + i, rate, momentum, decay);
    }
}

// Find the index of the maximum element in an array
// double *a: array
// int n: size of a, |a|
// returns: index of maximum element
int max_index(double *a, int n)
{
    if(n <= 0) return -1;
    int i;
    int max_i = 0;
    double max = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

// Calculate the accuracy of a model on some data d
// model m: model to run
// data d: data to run on
// returns: accuracy, number correct / total
double accuracy_model(model m, data d)
{
    matrix p = forward_model(m, d.X);
    int i;
    int correct = 0;
    for(i = 0; i < d.y.rows; ++i){
        if(max_index(d.y.data[i], d.y.cols) == max_index(p.data[i], p.cols)) ++correct;
    }
    return (double)correct / d.y.rows;
}

// Calculate the cross-entropy loss for a set of predictions
// matrix y: the correct values
// matrix p: the predictions
// returns: average cross-entropy loss over data points, 1/n Σ(-ylog(p))
double cross_entropy_loss(matrix y, matrix p)
{
    int i, j;
    double sum = 0;
    for(i = 0; i < y.rows; ++i){
        for(j = 0; j < y.cols; ++j){
            sum += -y.data[i][j]*log(p.data[i][j]);
        }
    }
    return sum/y.rows;
}


// Train a model on a dataset using SGD
// model m: model to train
// data d: dataset to train on
// int batch: batch size for SGD
// int iters: number of iterations of SGD to run (i.e. how many batches)
// double rate: learning rate
// double momentum: momentum
// double decay: weight decay
void train_model(model m, data d, int batch, int iters, double rate, double momentum, double decay)
{
    int e;
    for(e = 0; e < iters; ++e){
        data b = random_batch(d, batch);
        matrix p = forward_model(m, b.X);
        fprintf(stderr, "%06d: Loss: %f\n", e, cross_entropy_loss(b.y, p));
        matrix dL = axpy_matrix(-1, p, b.y); // partial derivative of loss dL/dy
        backward_model(m, dL);
        update_model(m, rate/batch, momentum, decay);
        free_matrix(dL);
        free_data(b);
    }
}


// Questions 
//
// 5.2.2.1 Why might we be interested in both training accuracy and testing accuracy? What do these two numbers tell us about our current model?// By training and testing, we can get whether the model is overfit, just-right, or underfit. // If both accuracies are low, it is likely underfit (model complexity is not high enough).// If training accuracy is high but testing is low, it is likely overfit. (The model is not generalized,  even fitting to the noisy data in the training model.)


// 5.2.2.2 Try varying the model parameter for learning rate to different powers of 10 (i.e. 10^1, 10^0, 10^-1, 10^-2, 10^-3) and training the model. What patterns do you see and how does the choice of learning rate affect both the loss during training and the final model accuracy?
// Too large learning rate makes learning overshooting with fluctuate learning error. // 10^0 gives the best test accuracy. (The smaller learning rate supposed to give good training // rates, little bit bad test score(overfitting))
// 10^1:  training: 0.8828, test:0.8828 (Error fluctuate --> Overshooting)// 10^0:  training: 0.9190, test:0.9159// 10^-1:  training: 0.904 , test: 0.9079// 10^-2:  training: 0.8593 , test: 0.8671// 10^-3:  training: 0.73295  , test: 0.7426

// 5.2.2.3 Try varying the parameter for weight decay to different powers of 10: (10^0, 10^-1, 10^-2, 10^-3, 10^-4, 10^-5). How does weight decay affect the final model training and test accuracy?
// Smaller weight decay helps better accuracy but it saturates in certain points. By adding a decay term, model complexity is penalized. // 10^0:  0.899/0.904// 10^-1: 0.903/0.907// 10^-2: 0.904/0.908// 10^-3: 0.904/ 0.908// 10^-4: 0.904/ 0.908// 10^-5: 0.904/ 0.908

// 5.2.3.1 Currently the model uses a logistic activation for the first layer. Try using a the different activation functions we programmed. How well do they perform? What's best?
// LRELU, RELU, Softmax and logistic activation show similar performance. //

// 5.2.3.2 Using the same activation, find the best (power of 10) learning rate for your model. What is the training accuracy and testing accuracy?
// Using Softmax, learning rate = 1// training accuracy: %f 0.9190333333333334// test accuracy:     %f 0.9159

// 5.2.3.3 Right now the regularization parameter `decay` is set to 0. Try adding some decay to your model. What happens, does it help? Why or why not may this be?
// decay = 10
//  training accuracy: %f 0.6214833333333334
// test accuracy:     %f 0.6221
// --> It shows underfitting. it penalizes too much on model complexity and generalizes too much.  
// 
// decay = 0.01
//training accuracy: %f 0.9187333333333333
//test accuracy:     %f 0.9159
// generalized on the training model but it gives proper representation on the test set. 

//// 5.2.3.4 Modify your model so it has 3 layers instead of two. The layers should be `inputs -> 64`, `64 -> 32`, and `32 -> outputs`. Also modify your model to train for 3000 iterations instead of 1000. Look at the training and testing error for different values of decay (powers of 10, 10^-4 -> 10^0). Which is best? Why?
// 10^-3 gives the best test accuracy. The nature of the model is complex, thus smaller decay does not cause overfitting until 10^3. // 10^-4 gives slightly small test accuracy while testing accuracy is high. It is likely to be overfitting at this point which needs more regularization. 
// 10^0  : 0.86805 / 0.8706 (training/test)// 10^-1 : 0.9172/ 0.9176// 10^-2 : 0.9243/ 0.9216// 10^-3:  0.9249/ 0.922// 10^-4:  0.9249/ 0.9218

// 5.3.2.1 How well does your network perform on the CIFAR dataset?
// It gives 82% accuracy. hyperparameter needs to be modified to find the better solution. 
// 

