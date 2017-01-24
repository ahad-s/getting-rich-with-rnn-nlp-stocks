	
#include <boost/array.hpp>
	
class RNN
{
private:
	const int featureDim;
	const int outputDim;
	const int ZERO_EPSILON = 0.0000000000000001;

	int M; // num examples
	int N; // num features
	int hiddenDim;
	int bpttLen;

	Eigen::MatrixXd maxX;
	Eigen::MatrixXd minX;
	Eigen::MatrixXd minMax;

	Eigen::MatrixXd thetaU;
	Eigen::MatrixXd thetaV;
	Eigen::MatrixXd thetaW;
	Eigen::MatrixXd X;
public:
	/* m will have columns:
	day
	month
	year
	hours
	minutes
	price0 (TODO, more prices like oil/FX rates/etc.)
	volume0 (^)
	(TODO: feature0 like volatility, etc.)
	*/

	/*
	M = # examples
	N = 2*(# stocks) + 1 (# articles sentiment)
	X = M x n
	*/

	// bpttLen = how far to look back, 30 = 30min
	RNN(Eigen::MatrixXd m, int len);
	~RNN() {};
	
	Eigen::MatrixXd getDateMatrix(); // first 5(? ish)
	Eigen::MatrixXd getPricesMatrix(); // last n cols

	Eigen::MatrixXd weightsRandomInit(int sizeR, int sizeC);
	void normalize(Eigen::MatrixXd &m);
	void denormalize(Eigen::MatrixXd &m);

	Eigen::MatrixXd activationState(Eigen::MatrixXd m);
	Eigen::MatrixXd activationOutput(Eigen::MatrixXd m);
	Eigen::MatrixXd activationGradientFunction(Eigen::MatrixXd &m);


	boost::array<Eigen::MatrixXd, 2> forwardProp(Eigen::MatrixXd x);
	boost::array<Eigen::MatrixXd, 3> backPropTT(Eigen::MatrixXd x, 
										Eigen::MatrixXd y);

	double cost(Eigen::MatrixXd &x,
						Eigen::MatrixXd &y,
						double reg_const = 0);

	Eigen::MatrixXd gradient(Eigen::MatrixXd &theta,
							Eigen::MatrixXd &x,
							Eigen::MatrixXd &y,
							double reg_const);

	Eigen::MatrixXd predict(Eigen::MatrixXd &x_test);

	void trainSGD(Eigen::MatrixXd &x_train,
								Eigen::MatrixXd &y_train,
								double alpha,
								int epoch = 5,
								int check_loss_after=1);

	bool gradChecking(double error = 0.01) {return true;};






};