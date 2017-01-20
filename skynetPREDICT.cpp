#include <iostream>
#include <vector>
#include <map>
#include <string>

#include <Eigen/Dense>

#include "skynetEXTRACT.h"

using namespace std;
using namespace Eigen;


int main(int argc, char const *argv[])
{
	string pre_stock = "USA500.IDX";
	string freq = "1_m"; // 1 minute bars
	string bidask = "BID"; // which price to display
	string date_from = "09.01.2017";
	string date_to = "13.01.2017";

	auto fname = [](string s1, string s2, string s3, 
				string s4, string s5){
		return s1 + "_Candlestick_" + s2 
		+ "_" + s3 + "_" + s4 + "-" + s5 + ".csv";
	};

	string csv_name = "resources/PRICES/" + 
						fname(pre_stock, freq, 
						bidask, date_from, date_to);

	// todo get this working lol...
	// Matrix3d tmp = skynetEXTRACT::getCSVData<Matrix3d>(csv_name);

	return 0;
}