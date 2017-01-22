
class RNN
{
private:
	const int inputDim;
	const int outputDim;

	int hiddenDim;
	int bpttLen;
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
	X = M x 
	*/
	// bpttLen = how far to look back, 30 = 30min
	RNN(Eigen::MatrixXd m, int bpttLen = 30);
	~RNN() {};
	
	Eigen::MatrixXd getDateMatrix(); // first 5(? ish)
	Eigen::MatrixXd getPricesMatrix(); // last n cols

	Eigen::MatrixXd weightsRandomInit(int sizeR, int sizeR);

	Eigen::MatrixXd activationFunction(Eigen::MatrixXd &m);
	Eigen::MatrixXd activationGradientFunction(Eigen::MatrixXd &m);

	vector<Eigen::MatrixXd> forwardProp(Eigen::MatrixXd x);
	vector<Eigen::MatrixXd> backPropTT(Eigen::MatrixXd x, 
										Eigen::MatrixXd y);

	Eigen::MatrixXd cost(Eigen::MatrixXd &theta,
						Eigen::MatrixXd &x,
						Eigen::MatrixXd &y,
						double reg_const);

	Eigen::MatrixXd gradient(Eigen::MatrixXd &theta,
							Eigen::MatrixXd &x,
							Eigen::MatrixXd &y,
							double reg_const);

	Eigen::MatrixXd predict(Eigen::MatrixXd &theta, 
							Eigen::MatrixXd &x_test);

	Eigen::MatrixXd SGDStep();
	Eigen::MatrixXd stochGradDesc(Eigen::MatrixXd &x_train,
								Eigen::MatrixXd &y_train,
								double alpha,
								int epoch,
								int check_loss_after=10);
	void train(int iter = 10);

	bool gradChecking(double error = 0.01) {return true;};






};