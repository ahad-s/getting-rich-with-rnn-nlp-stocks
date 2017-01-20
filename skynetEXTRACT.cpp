#include <skynetEXTRACT.h>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <string>

using namespace std;
using namespace Eigen;

/*
Usage:
Matrix(X/2/3/etc.)d M = getCSVData("potato.csv")
*/
template<typename Mat>
static Mat skynetEXTRACT::getCSVData(string &csvPath) const{
	ifstream inpData;
	inpData.open(csvPath);
	string line;
	vector<double> vals;
	uint rows = 0;

	while (getline(inpData, line)){
		stringstream lStream(line);
		string cell;

		while (getline(lStream, cell, ","))
			vals.push_back(stod(cell));

	}

	return Map<
				const Matrix<
						typename Mat::Scalar, 
						Mat::RowsAtCompileTime, 
						Mat::ColsAtCompileTime, 
						RowMajor
						>
				>(vals.data(), rows, values.size()/rows);

}