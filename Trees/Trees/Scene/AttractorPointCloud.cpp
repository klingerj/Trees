#include "AttractorPointCloud.h"

void AttractorPointCloud::GeneratePoints(unsigned int numPoints) {
    auto start = std::chrono::system_clock::now();
    for (unsigned int i = 0; i < numPoints; ++i) {
        const glm::vec3 p = glm::vec3(dis(rng) * 2.0f, dis(rng) * 2.0f, dis(rng) * 4.0f);
        points.emplace_back(AttractorPoint(points[i]));
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Elapsed time for Attractor Point Cloud Generation: " << elapsed_seconds.count() << "s\n";
    std::cout << "Number of Attractor Points Generated: " << points.size() << "\n\n";
}

void AttractorPointCloud::GeneratePoints(const Mesh& m, unsigned int numPoints) {
    auto start = std::chrono::system_clock::now();
    for (unsigned int i = 0; i < numPoints; ++i) {
        const glm::vec3 p = glm::vec3(dis(rng) * 2.0f, dis(rng) * 2.0f, dis(rng) * 4.0f);

        // Intersect with mesh instead
        if (m.Contains(p)) {
            points.emplace_back(AttractorPoint(points[i]));
        }
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Elapsed time for Attractor Point Cloud Generation: " << elapsed_seconds.count() << "s\n";
    std::cout << "Number of Attractor Points Generated: " << points.size() << "\n\n";
}