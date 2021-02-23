
#include <iostream>
#include <vector>
#include <sstream>

namespace machine_learning{

// Define the data structure and store the input data
class Point{
private:
    int m_pointId, m_clusterID;
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
        m_clusterID = 0;
    }

    int getDim(){
        return m_dim;
    }

    int getClusterId(){
        return m_clusterID;
    }

    void setClusterId(int id){
        m_pointId = id;
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
        m_clusterId = clusterId;
        for (int i = 0; i < centerPointId.getDim(); i++){
            m_centerPointId.push_back(centerPointId.getVal(i));
        }
        this->addPoint(centerPointId);
    }

    void addPoint(Point p){
        p.setClusterId(m_clusterId);
        m_points.push_back(p);
    }

    bool removePoint(int pointId){
        int size = m_points.size();
        for(int i=0; i<size; i++){
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

class KMeans{
private:
    int m_K, m_iters, m_dim, m_total_points;
    std::vector<Cluster> clusters;

    int getNearestClusterId(Point point);

public:
    KMeans(int k, int iters);
    void run(std::vector<Point> &all_points);
};

}
