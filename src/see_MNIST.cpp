// idk if this works I just copied gui and changed it a little
#include "read_MNIST.hpp"
#include "network.hpp"
#include "gui.hpp"

const int width = 280;
const int height = 280;
const char* window_name;
cv::Mat canvas;
cv::Rect quit_button;
cv::Rect grid[28][28];
double grid_on[28][28];
network test_net;

void draw_quit_button(){
	cv::rectangle(canvas, quit_button, cv::Scalar(255), cv::FILLED, cv::LINE_8);
	cv::putText(
		canvas(quit_button), "Quit",
		cv::Point(quit_button.width*0.3, quit_button.height*0.7), 
		cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0)
	);
}

void re_draw(){
	for(int i = 0; i < 28; i++){
		for(int j = 0; j < 28; j++){
			cv::rectangle(canvas, grid[i][j], cv::Scalar((int)grid_on[i][j]), cv::FILLED, cv::LINE_8);
		}
	}
}

void callback_func(int event, int x, int y, int flags, void* userdata){
	if(event == cv::EVENT_LBUTTONDOWN){
		if(quit_button.contains(cv::Point(x, y))){ run = false; }
	}
}

int main(int argc, char** argv){
	char* labels;
	char** images;
	int num_images = read_MNIST(TEST_IMAGES, TEST_LABELS, images, labels);

	nload(test_net);
	try{
		window_name = "Calebs Neural Network";
		quit_button = cv::Rect(cv::Point(width * 6 / 7, height + 20), cv::Point(width - 20, height + 50));
		canvas = cv::Mat::zeros(height + 20 + 60, width + 20, CV_8UC1);
		draw_quit_button();

		for(int i = 0; i < 28; i++){
			for(int j = 0; j < 28; j++){
				grid[i][j] = cv::Rect(cv::Point(i * 10 + 10, j * 10 + 10), cv::Point(i * 10 + 20, j * 10 + 20));
			}
		}

		cv::namedWindow(window_name);
		cv::setMouseCallback(window_name, callback_func);
		
		int temp = 0;
		for(int j = 0; j < 100; j++){
			for(int i = 0; i < 28; i++){
				for(int k = 0; k < 28; k++){
					grid_on[k][i] = 0;
					temp = images[j][k + i * 28];
					if(temp < 0){
						grid_on[k][i] = 255 + temp;
					} else if(temp > 0){
						grid_on[k][i] = temp - 255;
					}
					test_net.A0->elements[k + i * 28][0] = images[j][k + i * 28];
					std::cout << (int)images[j][k + i * 28] << " ";
				}
				std::cout << std::endl;
			}
			
			forward_prop(test_net);
			P("NUMBER: " << j);
			P("net guess: " << argmax(test_net));
			P("acutal: " << (int)labels[j]);
			P("-----");
			re_draw();
			cv::imshow(window_name, canvas);
			cv::waitKey(0);
		}
	} catch (const cv::Exception& e){
		const char* err_msg = e.what();
		P("Error " << err_msg);
	}
	
	/*** MEMORY CLEANUP ***/
	delete[] labels;
	for(int i = 0; i < num_images; i++){
		delete[] images[i];
	}
	delete[] images;
	nfree(test_net);
	return 0;
}
