#include <iostream>
#include <algorithm>
#include <list>
#include <vector>
#include <iterator>
#include <functional>
#include <time.h>
#include <chrono>
#include <cstdlib>
using namespace std;

struct triple
{
	long long int set;				//Set denotes which Elements are in the Subset
	long int w;						//Weight of the Triple				//d
	long int p;						//Profit of the Triple				//d

									//Struct Constructor
	triple() : set(0),
		w(0.0),
		p(0.0)
	{}

	//Comparison Operator Overloadings
	bool operator< (const triple &t) const
	{
		return (w < t.w);
	}

	bool operator> (const triple &t) const
	{
		return (w > t.w);
	}
};

void merge_lists(vector<triple> &A, vector<triple> &B, vector< pair<long long int, long long int> > &V)
{
	vector<triple> T_p, Tcopy;
	triple t;

	long long int v1s = V.size() >> 1, v2s = V.size() - v1s;

	//Initialisation for A
	t.set = 0, t.w = t.p = 0;
	A.push_back(t);

	//Sort A in Non-Increasing Order
	for (long long int i = 0; i < v1s; ++i)
	{
		T_p.clear();
		Tcopy.clear();

		//Add Elements to Subset (Triple) ti
		//Add ti to T_p
		for (long long int j = 0; j < (long long int)A.size(); ++j)
		{
			t.set = A[j].set + (1 << i);
			t.w = A[j].w + V[i].first;
			t.p = A[j].p + V[i].second;
			T_p.push_back(t);
		}

		//Merge A, T_p
		merge(A.begin(), A.end(), T_p.begin(), T_p.end(), back_inserter(Tcopy));
		A = Tcopy;
	}

	//Initialisation for B
	t.set = 0, t.w = t.p = 0;
	B.push_back(t);

	//Sort B in Non-Increasing Order
	for (long long int i = 0; i < v2s; ++i)
	{
		T_p.clear();
		Tcopy.clear();

		//Add Elements to Subset (Triple) ti
		//Add ti to T_p
		for (long long int j = 0; j < (long long int)B.size(); ++j)
		{
			t.set = B[j].set + (1 << i);
			t.w = B[j].w + V[i + v1s].first;
			t.p = B[j].p + V[i + v1s].second;

			T_p.push_back(t);
		}

		//Merge B, T_p
		merge(B.begin(), B.end(), T_p.begin(), T_p.end(), back_inserter(Tcopy), greater<struct triple>());
		B = Tcopy;
	}
}

void maxScan(vector<triple> &B, vector< pair<int, long int> > &maxB)
{
	long int Bsize = B.size();
	maxB[Bsize - 1].first = B[Bsize - 1].p;
	maxB[Bsize - 1].second = Bsize - 1;
	for (long int i = Bsize - 2; i >= 0; i--)
	{
		if (B[i].p>maxB[i + 1].first)
		{
			maxB[i].first = B[i].p;
			maxB[i].second = i;
		}
		else
		{
			maxB[i].first = maxB[i + 1].first;
			maxB[i].second = maxB[i + 1].second;
		}
	}
}

long int generate_sets(vector<triple> &A, vector<triple> &B, const int &c,
	vector< pair<int, long long int> > &maxB, long int N)
{
	int bestValue = 0;
	pair<long long int, long long int> bestSet;

	long long int i = 0, j = 0;
	while (i < N && j < N)
	{
		if (A[i].w + B[j].w > c)
		{
			++j;

			if (j == N) break;
			else continue;
		}

		if (A[i].p + maxB[j].first > bestValue)
		{
			bestValue = A[i].p + maxB[j].first;
			bestSet = make_pair(A[i].set, maxB[j].second);
		}
		++i;
	}

	return bestValue;
}

void dp_knapSack(long long int W, double wt[], double val[], long long int n)
{
	long long int i, w;
	vector< vector<double> > K(n + 1, vector<double>(W + 1));

	// Build table K[][] in bottom up manner 
	for (i = 0; i <= n; i++)
	{
		for (w = 0; w <= W; w++)
		{
			if (i == 0 || w == 0)
				K[i][w] = 0;
			else if (wt[i - 1] <= w)
				K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w]);
			else
				K[i][w] = K[i - 1][w];
		}
	}
	cout << "\n\n\tBest_DP: " << K[n][W] << endl;
}



//Input : Sorted Lists -> (A, B)
//Output : Partitioned Sorted Lists -> (Ak, Bk) with N/k elements each
void list_to_blocks(vector<triple> &A, vector<triple> &B, vector< vector<triple> > &Ak,
	vector< vector<triple> > &Bk, int k)
{
	long long int e = A.size() / k, i;

	vector<triple>::iterator Ait, Bit;
	Ait = A.begin(), Bit = B.begin();

//#pragma omp parallel for shared(A, B, Ak, Bk, e, k) private(i, Ait, Bit)
	for (i = 0; i < k; ++i)
	{
		Ait = A.begin() + i * e;
		Bit = B.begin() + i * e;

		copy(Ait, Ait + e, back_inserter(Ak[i]));
		copy(Bit, Bit + e, back_inserter(Bk[i]));
	}
}

//Input : Partitioned Sorted Lists -> (Ak, Bk)
//Output : Maximum Profit of Blocks -> (maxAi, maxBi)
void fsave_max_val(vector< vector<triple> > &Ak, vector< vector<triple> > &Bk,
	vector<double> &maxA, vector<double> &maxB)
{
	//Needs to be dynamic if not equally partitioned (if N/k not an int)
	long long int e = maxA.size(), i, j;
	double Amax, Bmax;

    #pragma omp parallel for shared(Ak, Bk,  maxA, maxB) private(e, i, j, Amax, Bmax) //Here
	for (i = 0; i < e; ++i)
	{
		Amax = Ak[i][0].p;
		Bmax = Bk[i][0].p;

		//Perform Parallel Max Search for Better Result
		for (j = 1; j < Ak[i].size(); ++j)
		{
			Amax = (Amax < Ak[i][j].p) ? Ak[i][j].p : Amax;
			Bmax = (Bmax < Bk[i][j].p) ? Bk[i][j].p : Bmax;
		}
		maxA[i] = Amax;
		maxB[i] = Bmax;
	}
}
    
//Input : Ak, Bk, maxAi, maxBi
//Output : Blocks that are within Capacity c
void prune(vector< vector<triple> > &Ak, vector< vector<triple> > &Bk, double c, vector<double> &maxA,
	vector<double> &maxB, vector< vector<int> > &candidate, double &bestValue)
{
	int Z, Y;
	int i, j, k = Ak.size(), e = Ak[0].size();
	vector<int> maxValue(k);

#pragma omp parallel for reduction(max:bestValue) shared(Ak, Bk, maxA, maxB,  maxValue, candidate) private(i, j, Z, Y,c, e)
	for (i = 0; i < k; ++i)
	{
		maxValue[i] = 0;
		for (j = 0; j < k; ++j)  //Here - will lead to CR
		{
			Z = Ak[i][0].w + Bk[j][e - 1].w;
			Y = Ak[i][e - 1].w + Bk[j][0].w;

			if (Y <= c)
			{
				if (maxA[i] + maxB[j] > maxValue[i])
					maxValue[i] = maxA[i] + maxB[j];

				if (bestValue<maxValue[i])    //Here
					bestValue = maxValue[i];
			}
			else if (Z <= c && Y > c)
				candidate[i].push_back(j); // here make copy of block bk[j]
		}
	}
}

//Input : Candidate Block Pairs -> candidate
//Output : (Max[i][j][t], L[j][t]) with reference to candidate[i]
void ssave_max_val(vector< vector<triple> > &Bk, vector< vector< vector< pair<double, long long int> > > > &Max,
	vector< vector<int> > &candidate, double &bestValue)
{
	int i, t, l, k = Bk.size();
	int j, e = Bk[0].size();

	#pragma omp parallel for shared(Bk,candidate,Max) private(i,j,t,l,k,e) //Here
	for (i = 0; i < k; ++i)
	{
		for (t = 0; t < candidate[i].size(); ++t)
		{
			//l is the Index of The Block Partition B of the Candidate Block Pair (Bk[l])
			l = candidate[i][t];
			//Initialise Last Element and Index
			Max[i][e - 1][t].first = Bk[l][e - 1].p;
			Max[i][e - 1][t].second = e - 1;

			//Reverse Inclusive Max-Scan
			for (j = e - 2; j > -1; --j)
			{
				if (Bk[l][j].p > Max[i][j + 1][t].first)
				{
					Max[i][j][t].first = Bk[l][j].p;
					Max[i][j][t].second = j;
				}
				else
				{
					Max[i][j][t].first = Max[i][j + 1][t].first;
					Max[i][j][t].second = Max[i][j + 1][t].second;
				}

			}
		}
	}
}

//Input : candidate, Max
//Output : Best Value
void par_search(vector< vector<triple> > &Ak, vector< vector<triple> > &Bk, double c, vector< vector<int> > &candidate,
	vector< vector< vector< pair<double, long long int> > > > &Max, double &bestValue)
{
	int i, j, t, l, k = Ak.size();
	long long int e = Ak[0].size(), X, Y;
	vector<double> maxValue(k);
	vector< pair<long long int, long long int> > Xi(k);

	//Xi -> (Index ID of Subset A, Index ID of Subset B)

#pragma omp parallel for shared(Ak, Bk, candidate, Max, Xi, maxValue) private(i, j, t, l, X, Y, e,k) //Here
	for (i = 0; i < k; ++i)
	{
		maxValue[i]=0; 
		Xi[i].first = 0, Xi[i].second = 0;// Here
		for (t = 0; t < candidate[i].size(); ++t)
		{
			l = candidate[i][t];
			X = 0, Y = 0;


			while (X < e && Y < e)
			{
				if (Ak[i][X].w + Bk[l][Y].w > c)
				{
					++Y;
					continue;
				}
				else if (Ak[i][X].p + Max[i][Y][t].first > maxValue[i])
				{
					maxValue[i] = Ak[i][X].p + Max[i][Y][t].first;

					Xi[i].first = Ak[i][X].set;
					Xi[i].second = Bk[l][Max[i][Y][t].second].set;
				}
				++X;
			}
		}
	}

	//Evaluate Maximum Profit from max(maxValue[i])
	long long int X1 = Xi[0].first, X2 = Xi[0].second;
	for (i = 0; i < k; ++i)
		if (bestValue < maxValue[i])
		{
			bestValue = maxValue[i];
			X1 = Xi[i].first;
			X2 = Xi[i].second;
		}

	//The Subset ID from A, Subset ID from B which gives Maximum Profit (Best Value)
	cout << "\n\tSubsets : " << X1 << ", " << X2 << endl;
	cout << "\tBestvalue : " << bestValue << endl;
}

int main()
{
	//Input Data
	int c = 0;
	vector< pair<long long int, long long int> > V;
	vector<double> wt_arr, p_arr;
	srand(time(0));

	//Number of Items
	int num_items = 10;

	//Input Data
	for (int i = 0; i < num_items; ++i)
	{
		double wt = rand() % (long int)1e7;
		double p = rand() % (long int)1e7;
		c += wt;
		V.push_back(make_pair(wt, p));
		wt_arr.push_back(wt);
		p_arr.push_back(p);
	}

	//Set capacity
	c /= 2;
	printf("\n\tCapacity = %d\n", c);

	//Computation & Timing
	auto start = chrono::steady_clock::now();

	/*
	[Ak, Bk]	 -> Ak has k Blocks with N/k elements each
	[maxA, maxB] -> maxI has one Element for each Block of List I
	[candidate] -> candidate[i] is a Vector of Blocks of Bk, which are candidate solutions with Ak[i]
	[Max[i][j][t], L[j][t]] -> Pair of Maximum Profit & Respective Index with reference to candidate[i]
	*/

	int k = 4;									//Number of Partitions
	long long int N = 1 <<( num_items >> 12);			//Number of Subsets
	long long int e = N / k;					//Number of Elements per Subset

	double bestValue = -1;			//d
	vector<triple> A, B;

	vector< vector<triple> > Ak(k, vector<triple>());
	vector< vector<triple> > Bk(k, vector<triple>());

	vector<double> maxA(k), maxB(k);
	vector< vector<int> > candidate(k);

	vector< vector< vector< pair<double, long long int> > > > Max(k, vector< vector< pair<double, long long int> > >(e, vector< pair<double, long long int>  >(2)));
	merge_lists(A, B, V);											//Currently Serial Merging

	list_to_blocks(A, B, Ak, Bk, k);								//Partition Lists to Blocks
	fsave_max_val(Ak, Bk, maxA, maxB);								//Save 
	prune(Ak, Bk, c, maxA, maxB, candidate, bestValue);
	ssave_max_val(Bk, Max, candidate, bestValue);
	//par_search(Ak, Bk, c, candidate, Max, bestValue);

	auto stop = chrono::steady_clock::now();

	cout << "\n  Computational Time (Parallel) : ";
	cout << (int)(chrono::duration_cast<chrono::nanoseconds>(stop - start).count()) / 1000000.0;
	cout << " ms" << endl;


	//Time the Serial DP Approach
	start = chrono::steady_clock::now();

	//dp_knapSack(c, &wt_arr[0], &p_arr[0], V.size());

	stop = chrono::steady_clock::now();

	cout << "\n  Computational Time (DP Serial) : ";
	cout << (int)(chrono::duration_cast<chrono::nanoseconds>(stop - start).count()) / 1000000.0;
	cout << " ms" << endl;
	cin.get();
	return 0;
}
