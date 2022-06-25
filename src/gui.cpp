#include "read_MNIST.hpp"
#include "network.hpp"
#include "gui.hpp"

bool run;
const int width = 280;
const int height = 280;
const char* window_name = "Calebs Neural Network";
bool mouse_down = false;
cv::Mat canvas;
cv::Rect clear_button;
cv::Rect guess_button;
cv::Rect quit_button;
cv::Rect grid[28][28];
double grid_on[28][28];
network test_net;

int get_rand(int min, int max){ return rand() % (max - min + 1) + min; }

void guess(){
		for(int i = 0; i < 28; i++){ for(int j = 0; j < 28; j++){ test_net.A0->elements[j + i * 28][0] = grid_on[j][i]; } }
		forward_prop(test_net);
		P("Network guess is: " << argmax(test_net));
}

void draw_clear_button(){
	cv::rectangle(canvas, clear_button, cv::Scalar(255), cv::FILLED, cv::LINE_8);
	cv::putText(
		canvas(clear_button), "Clear",
		cv::Point(clear_button.width*0.3, clear_button.height*0.7), 
		cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0)
	);
}

void draw_guess_button(){
	cv::rectangle(canvas, guess_button, cv::Scalar(255), cv::FILLED, cv::LINE_8);
	cv::putText(
		canvas(guess_button), "guess",
		cv::Point(guess_button.width*0.3, guess_button.height*0.7), 
		cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0)
	);
}

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
	if(event == cv::EVENT_MOUSEMOVE && mouse_down){ 
		for(int i = 0; i < 28; i++){
			for(int j = 0; j < 28; j++){
				if(grid[i][j].contains(cv::Point(x, y))){
					grid_on[i][j] = 255;
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
		if(clear_button.contains(cv::Point(x, y))){
			for(int i = 0; i < 28; i++){ for(int j = 0; j < 28; j++){ grid_on[i][j] = 0; } }
			re_draw();
		} else if(guess_button.contains(cv::Point(x, y))){ 
			guess();
		} else if(quit_button.contains(cv::Point(x, y))){
			run = false;
		} mouse_down = true;
	} else if(event == cv::EVENT_LBUTTONUP){ 
		mouse_down = false;
	}
}

int main(int argc, char** argv){
	nload(test_net);
	try{
		clear_button = cv::Rect(cv::Point(width / 11, height + 20), cv::Point(width / 11 + 90, height + 50));
		guess_button = cv::Rect(cv::Point(width * 3 / 7, height + 20), cv::Point(width * 3 / 7 + 90, height + 50));
		quit_button = cv::Rect(cv::Point(width * 3 / 7 + 100, height + 20), cv::Point(width * 3 / 7 + 160, height + 50));
		canvas = cv::Mat::zeros(height + 20 + 60, width + 20, CV_8UC1);
		draw_clear_button(); draw_guess_button(); draw_quit_button();
		cv::namedWindow(window_name);
		cv::setMouseCallback(window_name, callback_func);

		for(int i = 0; i < 28; i++){
			for(int j = 0; j < 28; j++){
				grid[i][j] = cv::Rect(cv::Point(i * 10 + 10, j * 10 + 10), cv::Point(i * 10 + 20, j * 10 + 20));
			}
		}

		run = true;
		while(cv::waitKey(20) != 27 && run){
			cv::imshow(window_name, canvas);
		}
	} catch (const cv::Exception& e){
		const char* err_msg = e.what();
		P("Error " << err_msg);
	}
	nfree(test_net);
	return 0;
}
