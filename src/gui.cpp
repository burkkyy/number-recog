#include "read_MNIST.hpp"
#include "network.hpp"
#include "gui.hpp"

const int width = 280;
const int height = 280;
const char* window_name;
bool mouse_down = false;
cv::Mat canvas;
cv::Rect clear_button;
cv::Rect guess_button;
cv::Rect grid[28][28];
int grid_on[28][28];
network net;

int get_rand(int min, int max){
	return rand() % (max - min + 1) + min;
}

void draw_clear_button(){
	cv::rectangle(
		canvas,
		clear_button,
		cv::Scalar(255),
		cv::FILLED,
		cv::LINE_8
	);
	cv::putText(
		canvas(clear_button), 
		"Clear", 
		cv::Point(clear_button.width*0.3, clear_button.height*0.7), 
		cv::FONT_HERSHEY_PLAIN, 1, 
		cv::Scalar(0)
	);
}

void draw_guess_button(){
	cv::rectangle(
		canvas,
		guess_button,
		cv::Scalar(255),
		cv::FILLED,
		cv::LINE_8
	);
	cv::putText(
		canvas(guess_button), 
		"guess", 
		cv::Point(guess_button.width*0.3, guess_button.height*0.7), 
		cv::FONT_HERSHEY_PLAIN, 1, 
		cv::Scalar(0)
	);
}

void re_draw(){
	for(int i = 0; i < 28; i++){
		for(int j = 0; j < 28; j++){
			cv::rectangle(
				canvas,
				grid[i][j],
				cv::Scalar((int)grid_on[i][j]),
				cv::FILLED,
				cv::LINE_8
			);
		}
	}
}

void callback_func(int event, int x, int y, int flags, void* userdata){
	if(event == cv::EVENT_MOUSEMOVE && mouse_down){ 
		// P("drawing");
		for(int i = 0; i < 28; i++){
			for(int j = 0; j < 28; j++){
				if(grid[i][j].contains(cv::Point(x, y))){
					grid_on[i][j] = get_rand(253, 255);
					// draw around grid[i][j]
					if(0 < i < 28 && 0 < j < 28){
						if(grid_on[i-1][j-1] < grid_on[i][j]){ grid_on[i-1][j-1] = get_rand(90, 140); }
						if(grid_on[i][j-1] < grid_on[i][j]){ grid_on[i][j-1] = get_rand(90, 140); }
						if(grid_on[i+1][j-1] < grid_on[i][j]){ grid_on[i+1][j-1] = get_rand(90, 140); }
						if(grid_on[i+1][j] < grid_on[i][j]){ grid_on[i+1][j] = get_rand(90, 140); }
						if(grid_on[i+1][j+1] < grid_on[i][j]){ grid_on[i+1][j+1] = get_rand(90, 140); }
						if(grid_on[i][j+1] < grid_on[i][j]){ grid_on[i][j+1] = get_rand(90, 140); }
						if(grid_on[i-1][j+1] < grid_on[i][j]){ grid_on[i-1][j+1] = get_rand(90, 140); }
						if(grid_on[i-1][j] < grid_on[i][j]){ grid_on[i-1][j] = get_rand(90, 140); }
					}
				}
			}
		}
		re_draw();
	} else if(event == cv::EVENT_LBUTTONDOWN){
		mouse_down = true;
		if(clear_button.contains(cv::Point(x, y))){
			for(int i = 0; i < 28; i++){
				for(int j = 0; j < 28; j++){
					grid_on[i][j] = 0;
				}
			}
			re_draw();
		}
		if(guess_button.contains(cv::Point(x, y))){
			for(int i = 0; i < 28; i++){
				for(int j = 0; i < 28; i++){
					net.A0->elements[j + 28 * i][0] = (double)grid_on[i][j];
				}
			}
			forward_prop(net);
			//print_output(net);
			for(int i = 0; i < 10; i++){
				P(net.B2->elements[i][0]);
			}
			int guess = argmax(net);
			P("guess is: " << guess);
		}
	} else if(event == cv::EVENT_LBUTTONUP){	
		mouse_down = false;
	}

	cv::imshow(window_name, canvas);
	cv::waitKey(1);
}

int main(int argc, char** argv){
	char* labels;
	char** images;
	int num_images = read_MNIST(TEST_IMAGES, TEST_LABELS, images, labels);
	nload(net);

	try{
		window_name = "Calebs Neural Network";
		clear_button = cv::Rect(cv::Point(width / 6, height), cv::Point(width / 2, height + 30));
		guess_button = cv::Rect(cv::Point(width * 4 / 7, height), cv::Point(width - 20, height + 30));
		canvas = cv::Mat::zeros(width + 30, height, CV_8UC1);
		draw_clear_button();
		draw_guess_button();
		
		for(int i = 0; i < 28; i++){
			for(int j = 0; j < 28; j++){
				grid[i][j] = cv::Rect(
								cv::Point(i * 10, j * 10),
								cv::Point(i * 10 + 10, j * 10 + 10)
							);
			}
		}

		cv::namedWindow(window_name);
		cv::setMouseCallback(window_name, callback_func);
		while(cv::waitKey(20) != 27){
			cv::imshow(window_name, canvas);
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

	return 0;
}

