
#include <iomanip>
#include <iostream>
#include <vector>
using namespace std;

#include "init.h"
#include "main.h"
#include "ref.h"
#include "types.h"

#include "test_kNN_CUDA.h"

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

    // basic arguments readout
    if (argc == 5)
    {
        trainSize = atoi(argv[1]);
        testSize = atoi(argv[2]);
        DIM = atoi(argv[3]);
        K = atoi(argv[4]);
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
    srand(time(NULL));
    init(&trainPoints[0], trainSize * DIM, 0.0f, 10.0f);
    init(&testPoints[0], testSize * DIM, 0.0f, 10.0f);

    // Compute the ground truth
    Ref_kNN(trainPoints, testPoints, DIM, K, gt_distances, gt_indices);

    // for some reason kNN-CUDA works with DIM & K as leading dimensions
    // rows: DIM & K
    // cols: trainSize & testSize
    vector<float> trainPointsTr = Transpose(trainPoints, trainSize, DIM);
    vector<float> testPointsTr = Transpose(testPoints, testSize, DIM);
    vector<float> gt_distancesTr = Transpose(gt_distances, testSize, K);
    vector<int> gt_indicesTr = Transpose(gt_indices, testSize, K);

    // Display header
    int width = 16;
    cout << setw(width) << "Implementation";
    cout << setw(width) << "Duration (s)";
    cout << setw(width) << "Nr. iterations";
    cout << setw(width) << "Validation";
    cout << endl;

    // Test and display results
    test_kNN_CUDA(trainPointsTr, testPointsTr, DIM, K, gt_distancesTr, gt_indicesTr, &knn_cuda_global,  "knn_cuda_global",  100);
    test_kNN_CUDA(trainPointsTr, testPointsTr, DIM, K, gt_distancesTr, gt_indicesTr, &knn_cuda_texture, "knn_cuda_texture", 100);
    test_kNN_CUDA(trainPointsTr, testPointsTr, DIM, K, gt_distancesTr, gt_indicesTr, &knn_cublas,       "knn_cublas",       100);

    cout << endl;

    return EXIT_SUCCESS;
}
