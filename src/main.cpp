
#include <cstring>

#include <iomanip>
#include <iostream>
#include <vector>
using namespace std;

#include "init.h"
#include "main.h"
#include "ref.h"
#include "report.h"
#include "types.h"

#include "test_kNN_CUDA.h"
#include "test_flann.h"

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    cout << endl;

    // default arguments
    int trainSize = 128;
    int testSize  = 8;
    int DIM       = 3;
    int K         = 4;
    bool validation = true;

    // basic arguments readout
    if (5 <= argc)
    {
        trainSize = atoi(argv[1]);
        testSize = atoi(argv[2]);
        DIM = atoi(argv[3]);
        K = atoi(argv[4]);

        if (6 <= argc)
            validation = 0 != strcmp("-v", argv[5]);
    }

    if (trainSize < K)
    {
        cout << "Error: K must be smaller than the number of training points." << endl;

        return 1;
    }

    // Display arguments
    cout << "Training points : " << trainSize << endl;
    cout << "Testing points  : " << testSize << endl;
    cout << "Dimension       : " << DIM << endl;
    cout << "K               : " << K << endl;
    cout << endl;

    // Allocate input points and output K-NN distances / indexes
    vector<float> trainPoints(trainSize * DIM);
    vector<float> testPoints(testSize * DIM);
    vector<float> gt_distances(testSize * K);
    vector<int>   gt_indices(testSize * K);

    // Initialize train & test points
    // srand(time(NULL));
    init(&trainPoints[0], trainSize * DIM, 0.0f, 10.0f);
    init(&testPoints[0], testSize * DIM, 0.0f, 10.0f);

    // Compute the ground truth
    if (validation)
        Ref_kNN(trainPoints, testPoints, DIM, K, gt_distances, gt_indices);

    DisplayHeader(validation);

    // Test and display results
    test_kNN_CUDA(trainPoints, testPoints, DIM, K, gt_distances, gt_indices, 100, validation);
    test_flann(trainPoints, testPoints, DIM, K, gt_distances, gt_indices, "flann", 100, validation);

    cout << endl;

    return EXIT_SUCCESS;
}
