#include <iostream>
#include <algorithm>
#include <cmath>
#include "Kmeans.cpp"

using namespace machine_learning;

int main(int argc, char **argv)
{
    if(argc != 3){
        std::cout << "Error: command-line argument count mismatch." << std::endl;
        return 1;
    }

    // number of clusters
    int K = atoi(argv[2]);
    std::string filename = argv[1];
    std::ifstream infile(filename.c_str());

    //Fetching points from file
    int pointId = 1;
    std::vector<machine_learning::KMeans::Point> all_points;
    std::string line;

    while(getline(infile, line)){
        machine_learning::KMeans::Point point(pointId, line);
        all_points.push_back(point);
        pointId++;
    }
    infile.close();
    std::cout << "\nData fetched successfully!" << std::endl;

    //Return if number of clusters > number of points
    if(all_points.size() < K){
        std::cout << "Error: Number of clusters greater than number of points." << std::endl;
        return 1;
    }

    //Running K-Means Clustering
    int iters = 100;

    machine_learning::KMeans::Kmeans k_means(K, iters);
    k_means.run(all_points);

    return 0;
    return 0;
}
