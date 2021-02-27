#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cmath>


namespace machine_learning{
namespace KMeans{

// Define the data structure and store the input data
class Point{
private:
    int m_pointId, m_clusterId;
    int m_dim;
    std::vector<double> m_values;

public:
    Point(int id, std::string input_line){
        m_pointId = id;
        m_dim = 0;
        // Can be used to split character strings separated by spaces, tabs and other symbols
        std::stringstream is(input_line);
        double val;
        while (is >> val){
            m_values.push_back(val);
            m_dim++;
        }
        m_clusterId = 0;
    }

    int getDim(){
        return m_dim;
    }

    int getClusterId(){
        return m_clusterId;
    }

    void setClusterId(int id){
        m_clusterId = id;
    }

    int getPointId(){
        return m_pointId;
    }

    double getVal(int pos){
        return m_values[pos];
    }

    double getId(){
        return m_pointId;
    }
};

class  Cluster{
private: 
    int m_clusterId;
    // Center point id
    std::vector<double> m_centerPointId;
    std::vector<Point> m_points;

public:
    Cluster(int clusterId, Point centerPointId){
        this->m_clusterId = clusterId;
        for (int i = 0; i < centerPointId.getDim(); i++){
            this->m_centerPointId.push_back(centerPointId.getVal(i));
        }
        this->addPoint(centerPointId);
    }

    void addPoint(Point p){
        p.setClusterId(this->m_clusterId);
        m_points.push_back(p);
    }

    bool removePoint(int pointId){
        int size = m_points.size();
        for (int i = 0; i < size; i++){
            if(m_points[i].getId()==pointId){
                m_points.erase(m_points.begin() + i);
                return true;
            }
        }
        return false;
    }

    int getId(){
        return m_clusterId;
    }

    Point getPoint(int pos){
        return m_points[pos];
    }

    int getSize(){
        return m_points.size();
    }

    double getCenterIdByPos(int pos){
        return m_centerPointId[pos];
    }

    void setCenterIdByPos(int pos, double val){
        this->m_centerPointId[pos] = val;
    }
};

class Kmeans{
private:
    int m_K, m_iters, m_dim, m_total_points;
    std::vector<Cluster> m_clusters;

    int getNearestClusterId(Point point){
        double sum = 0.0, min_dist;
        int NearestClusterId;

        for (int i = 0; i < m_dim; i++)
            sum += pow(m_clusters[0].getCenterIdByPos(i) - point.getVal(i), 2.0);
        min_dist = sqrt(sum);

        NearestClusterId = m_clusters[0].getId();

        for (int i = 1; i < m_K; i++){
            double dist;
            sum = 0.0;
            for (int j = 0; j < m_dim; j++)
                sum += pow(m_clusters[i].getCenterIdByPos(j) - point.getVal(j), 2.0);
            dist = sqrt(sum);

            if(dist < min_dist){
                min_dist = dist;
                NearestClusterId = m_clusters[i].getId();
            }
        }
        return NearestClusterId;
    }

public:
    Kmeans(int k, int iters){
        this->m_K = k;
        this->m_iters = iters;
    }
    void run(std::vector<Point> &input_points){
        m_total_points = input_points.size();
        m_dim = input_points[0].getDim();

        // Initializing Clusters;
        std::vector<int> used_pointIds;

        for (int i = 1; i <= m_K; i++){
            while (true){
                int index = rand() % m_total_points;
                if(find(used_pointIds.begin(), used_pointIds.end(), index)==used_pointIds.end()){
                    used_pointIds.push_back(index);
                    input_points[index].setClusterId(i);
                    Cluster cluster(i, input_points[index]);
                    m_clusters.push_back(cluster);
                    break;
                }
            }
        }
        std::cout << "Clusters initialized = " << m_clusters.size() << std::endl;
        std::cout << "Running K-Means Clustering..." << std::endl;

        int iter = 1;
        while (true){
            std::cout << "Iter: " << iter << "/" << m_iters << std::endl;
            bool done;

            // add input_points
            for (int i = 0; i < m_total_points; i++){
                int currentClusterId = input_points[i].getClusterId();
                int nearestClusterId = getNearestClusterId(input_points[i]);

                if(currentClusterId != nearestClusterId){
                    if(currentClusterId != 0){
                        for (int j = 0; j < m_K; j++){
                            if(m_clusters[j].getId()==currentClusterId){
                                m_clusters[j].removePoint(input_points[i].getId());
                            }
                        }
                    }
                    for (int j = 0; j < m_K; j++){
                        if(m_clusters[j].getId()== nearestClusterId){
                            m_clusters[j].addPoint(input_points[i]);
                        }
                    }
                    input_points[i].setClusterId(nearestClusterId);
                    done = false;
                }
            }

            // recalculating the center of each cluster
            for (int i = 0; i < m_K; i++){
                int ClusterSize = m_clusters[i].getSize();
                for (int j = 0; j < m_dim; j++){
                    double sum = 0.0;
                    if(ClusterSize > 0){
                        for (int p = 0; p < ClusterSize; p++){
                            sum += m_clusters[i].getPoint(p).getVal(j);
                        }
                        m_clusters[i].setCenterIdByPos(j, sum / ClusterSize);
                    }
                }
            }
            if(done||iter>=m_iters){
                std::cout << "Clustering completed in iteration : " << iter << std::endl;
                break;
            }
            iter++;
        }
        
        // print PointIds in each cluster
        for (int i = 0; i < m_K; i++){
            std::cout << "Points in cluster " << m_clusters[i].getId() << " : ";
            for (int j = 0; j < m_clusters[i].getSize(); j++){
                std::cout << m_clusters[i].getPoint(j).getId() << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "========================" << std::endl;

        //Write cluster centers to file
        std::ofstream outfile;
        outfile.open("clusters.txt");
        if(outfile.is_open()){
            for(int i=0; i<m_K; i++){
                std::cout<<"Cluster "<<m_clusters[i].getId()<<" centroid : ";
                for (int j = 0; j < m_dim; j++){
                    std::cout<<m_clusters[i].getCenterIdByPos(j)<<" ";     //Output to console
                    outfile<<m_clusters[i].getCenterIdByPos(j)<<" ";  //Output to file
                }
                std::cout<<std::endl;
                outfile<<std::endl;
            }
            outfile.close();
        }
        else{
            std::cout<<"Error: Unable to write to clusters.txt";
        }
    }
};

}
}
