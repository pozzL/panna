//implementatin of crNN by pozzL

#include <highfive/H5Easy.hpp>
#include <chrono>
#include <filesystem>

#include "panna/data.hpp"
#include "panna/distance.hpp"
#include "panna/lsh/euclidean.hpp"
#include "panna/rand.hpp"
#include "panna/trieindex.hpp"

float compute_recall( const std::vector<std::pair<float, uint32_t>>& ground,
                      const std::vector<std::pair<float, uint32_t>>& actual ) {
    size_t k = actual.size();
    float thresh = ground[k - 1].first;

    float cnt = 0.0;
    for ( auto pair : actual ) {
        if ( pair.first <= thresh ) {
            cnt += 1.0;
        }
    }

    return cnt / k;
}

int main(int, char** ) {

    //instances of k and L
    // 1) i have to calculate w K and L here for then instantiating the objects
    // nad this sould occur in a loop that ends when R is diameter of dataset
    // 2) i should pass the builder w and dimensions skippint the evaluation of it
    // 3)  then i should fix in a wat that i can use template knowing k in runtime
    using Distance = panna::EuclideanDistance;
    using Dataset = panna::EuclideanPoints;
    using HasherBuilder = panna::E2LSHBuilder<K, Dataset, Distance>;
    using Hasher = HasherBuilder::Output;

    H5Easy::File file( "fashion-mnist-784-euclidean.hdf5", 
                      H5Easy::File::ReadOnly );
    

    std::vector<std::vector<float>> data =
        H5Easy::load<std::vector<std::vector<float>>>( file, "/train" );

    std::vector<std::vector<float>> queries = 
        H5Easy::load<std::vector<std::vector<float>>>( file, "/test" );


    queries.resize( 1000 ); 

    std::cout << "data loaded" << std::endl;

    size_t dimensions = data[0].size();

    HasherBuilder hbuilder (dimensions);

    panna::Index<Dataset, Hasher, Distance> index( dimensions, hbuilder, //L//);
    for ( auto& p : data ) {
        index.insert( p.begin(), p.end() );
    }
    index.rebuild();

    // 4) then i need to add a new search method instead of the exploring trie one
    // that is currently being used, just looking in the buckets where q is in
    // and checking if a point within c*R is found for the cR-NN.

}
