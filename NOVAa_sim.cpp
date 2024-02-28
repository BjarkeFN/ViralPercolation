#include <iostream>
#include <fstream>
#include <armadillo>
#include <experimental/random>
#include <algorithm>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/filesystem.hpp>
#include <sstream>
#include <ctime>
#include <cmath>
#include <bitset>
#include <stack>
#include <unistd.h> // Unix-specific, for sleep() etc

using namespace std;
using namespace arma;

struct Params {
	int L; // Side-length of lattice
	float p_A; // Initial fraction in A state
	int T; // Number of timesteps to run
	int R_N; // Interferon radius
	bool COVID; // Use COVID rules or not
};

struct simdata {
    // States: 0=O, 1=V, 2=a, 3=A, 4=N
    // Rules:
    // 4(2) = 3
    // 4(0) = 2
    // 1(2) = 4
    // 1(0) = 1
	map<int, umat> Slat_t;
	Params parameters;
};

// Function for returning strings of floats with higher precision
template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

// Function for hashing, to get a "more unique" seed.
// http://www.concentric.net/~Ttwang/tech/inthash.htm
unsigned long mix(unsigned long a, unsigned long b, unsigned long c)
{
    a=a-b;  a=a-c;  a=a^(c >> 13);
    b=b-c;  b=b-a;  b=b^(a << 8);
    c=c-a;  c=c-b;  c=c^(b >> 13);
    a=a-b;  a=a-c;  a=a^(c >> 12);
    b=b-c;  b=b-a;  b=b^(a << 16);
    c=c-a;  c=c-b;  c=c^(b >> 5);
    a=a-b;  a=a-c;  a=a^(c >> 3);
    b=b-c;  b=b-a;  b=b^(a << 10);
    c=c-a;  c=c-b;  c=c^(b >> 15);
    return c;
}

void save_parameters(Params parameters, string datadir) {
	ofstream parameters_file;
	parameters_file.open(datadir + "parameters.dat");
	string outstr = "";
	outstr.append("L");
	outstr.append(" ");
	outstr.append(to_string(parameters.L));
	outstr.append("\n");
	outstr.append("T");
	outstr.append(" ");
	outstr.append(to_string(parameters.T));
	outstr.append("\n");
	outstr.append("p_A");
	outstr.append(" ");
	outstr.append(to_string(parameters.p_A));
	outstr.append("\n");
	outstr.append("R_N");
	outstr.append(" ");
	outstr.append(to_string(parameters.R_N));
	outstr.append("\n");
	outstr.append("COVID");
	outstr.append(" ");
	outstr.append(to_string(parameters.COVID));
	outstr.append("\n");
	parameters_file << outstr;
	parameters_file.close();
}

vector<int> s(int a, int b) {
	vector<int> out;
	out.push_back(a);
	out.push_back(b);
	return out;
}

int fix_neg(int n, int L) {
	int out;
	if (n < 0) {
		out = L+n;
	}
	else {
		out = n;
	}
	return out;
}

vector<vector<int>> get_neighbours_radius(int x, int y, int L, int R) {
	vector<vector<int>> nbors;
	vector<int> nb;
	int x_max;
	int y_max;
	int x_min;
	int y_min;
	x_max = x + R;
	y_max = y + R;
	x_min = x - R;
	y_min = y - R;
	int d;
	for (int x_p = x_min; x_p <= x_max; x_p++) {
		for (int y_p = y_min; y_p <= y_max; y_p++) {
			d = (x_p-x)*(x_p-x) + (y_p-y)*(y_p-y); 
			if (d==0) {
				// Skip
			}
			else if (d <= R*R) {
				nb.clear();
				nb.push_back(fix_neg(x_p % L,L));
				nb.push_back(fix_neg(y_p % L,L));
				nbors.push_back(nb);
			}
		}
	}
	//cout << "Neighbours: " << nbors.size() << endl;
	return nbors;
}

int main(int argc, char **argv) {

    bool save_every;
    save_every = false;
    string init_cond = "non-A"; // "random", "A" or "non-A"

	unsigned long seed = mix(clock(), time(NULL), getpid());
	srand(seed);
	Params parameters;
	parameters.L = 100;
	parameters.T = 100;
	parameters.COVID = true;
	parameters.p_A = 0.1;

	if (argc > 1) {
		parameters.L = stoi(argv[1]);
		cout << "L (passed via command line) = " << parameters.L << endl;
	}
	if (argc > 2) {
		parameters.T = stoi(argv[2]);
		cout << "T (passed via command line) = " << parameters.T << endl;
	}
	if (argc > 3) {
		parameters.p_A = stof(argv[3]);
		cout << "p_A (passed via command line) = " << parameters.p_A << endl;
	}
	if (argc > 4) {
		parameters.R_N = stoi(argv[4]);
		cout << "R_N (passed via command line) = " << parameters.R_N << endl;
	}
	if (argc > 5) {
		int COVID_int = stoi(argv[5]);
		if (COVID_int==1) {
			parameters.COVID = true;
		}
		else {
			parameters.COVID = false;
		}
		cout << "COVID (passed via command line) = " << parameters.COVID << endl;
	}

	int T = parameters.T;
	int L = parameters.L;
	float p_A = parameters.p_A;
	bool COVID = parameters.COVID;
	int updates_per_timestep = L*L;

	string datadir = "./data/data_p_" + to_string(int(round(100000*p_A))) + "_" + "R_" + to_string(parameters.R_N) + "_" + to_string(seed) + "/";
	cout << "Data directory: " << datadir << endl;
	boost::filesystem::create_directories(datadir);

	
	umat Slat = umat(L, L, fill::zeros);
	uvec NAR = uvec(T, fill::zeros); // "Attack rate" of N state
	uvec OAR = uvec(T, fill::zeros); // "Attack rate" of O state
	uvec VAR = uvec(T, fill::zeros); // "Attack rate" of V state
	uvec AAR = uvec(T, fill::zeros); // "Attack rate" of A state
	uvec aAR = uvec(T, fill::zeros); // "Attack rate" of a state


	int n;
	int m;
	int state;
	vector<int> nb = s(0,0); // Neighbour
	vector<vector<int>> neighbours;

	// Set initially antiviral
	for (int m=0; m < L; m++) {
		for (int n=0; n < L; n++) {
			if (randu() < p_A) {
				Slat(n,m) = 2;
			}
		}
	}

	// Set initial infected:
	// Random starting position:
	//n = rand() % L;
	//m = rand() % L;
	// Fixed starting position:
	n = int(L/2.0);
	m = int(L/2.0);
	n = fix_neg(n, L);
	m = fix_neg(m, L);
	Slat(n,m) = 1;

	int n_initial = n; // For later use
	int m_initial = m; // For later use

	int AR_prev = 0;
	int AR_curr = 0;
	int t_unchanged = 0;
	
	bool first_skip = true;

	for (int t=0; t < T; t++) {
		if (t > 0) {
			AR_prev = AR_curr;
		}
		else {
			AR_prev = accu(Slat==1)+accu(Slat==4);
		}
		if (t_unchanged < 10) {
			for (int i=0; i < updates_per_timestep; i++) {
				n = fix_neg(rand() % L,L);
				m = fix_neg(rand() % L,L);
				state = Slat(n,m);
				if (state == 1) { // V state
					neighbours =  get_neighbours_radius(n, m, L, 1);
					for (int nbidx = 0; nbidx < neighbours.size(); nbidx++) {
						nb = neighbours[nbidx];
						if (Slat(nb[0],nb[1])==0) { // Neighbour is naive cell
						    Slat(nb[0],nb[1]) = 1; // Put into V state
						}
						else if (Slat(nb[0],nb[1])==2) { // Neighbour in 'a' state
						    Slat(nb[0],nb[1]) = 4; // Put into N state
						}
					}
				}
				else if (state == 4) { // N state
					neighbours =  get_neighbours_radius(n, m, L, parameters.R_N);
					for (int nbidx = 0; nbidx < neighbours.size(); nbidx++) {
						nb = neighbours[nbidx];
						if (Slat(nb[0],nb[1]) == 0) { // Neighbour is naive cell
							Slat(nb[0],nb[1]) = 2; // Put into 'a' state
						}
						else if (Slat(nb[0],nb[1])==2) { // Neighbour in 'a' state 
							Slat(nb[0],nb[1]) = 3; // Put into A state
						}
					}
				}
			}
			NAR(t) = accu(Slat==4);
			OAR(t) = accu(Slat==0);
			VAR(t) = accu(Slat==1);
			AAR(t) = accu(Slat==3);
			aAR(t) = accu(Slat==2);
			AR_curr = accu(Slat==1)+accu(Slat==4);
		}
		else {
			if (first_skip) {
				cout << "Skipping timestep " << t << ", nothing changed." << endl;
				first_skip = false;
			}
			NAR(t) = NAR(t-1);
			OAR(t) = OAR(t-1);
			VAR(t) = VAR(t-1);
			AAR(t) = AAR(t-1);
			aAR(t) = aAR(t-1);
			AR_curr = NAR(t) + VAR(t);
		}
    	
		// States: 0=O, 1=V, 2=a, 3=A, 4=N
		if ((t == 10) or (t == T-1)) {
			cout << "Attack rate at t=" << t << ": " << AR_curr << ", fractional: " << AR_curr/((float) (parameters.L*parameters.L)) << endl;
		}
        if (save_every && t_unchanged < 5) {
    		Slat.save(datadir + "Slat_" + to_string(t) + ".dat", csv_ascii);
        }
        else {
            if (t == T-1) {
				// Commented out to save space:
                //Slat.save(datadir + "Slat_" + to_string(t) + ".dat", csv_ascii);
                }
            }
		if (AR_curr == AR_prev) {
			t_unchanged++;
		}
		else {
			t_unchanged = 0;
		}
	}
	cout << "Saving parameters." << endl;
	save_parameters(parameters, datadir);
	cout << "Saving attack rates." << endl;
	NAR.save(datadir + "NAR.dat", csv_ascii);
	OAR.save(datadir + "OAR.dat", csv_ascii);
	VAR.save(datadir + "VAR.dat", csv_ascii);
	AAR.save(datadir + "AAR.dat", csv_ascii);
	aAR.save(datadir + "aAR.dat", csv_ascii);
	cout << "Programme done." << endl;
	return 0;
}
