/*
 * particle_filter.cpp
 *
 *  Created on: Sep 03, 2017
 *  Author: Ricardo Zuccolo
 */

#include <random> //default_random_engine, normal_distribution
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles	
	num_particles = 10;

	// Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	const double init_weight = 1.0;

	for (int i = 0; i < num_particles; ++i) {
		particles.push_back(Particle {i, dist_x(gen), dist_y(gen), dist_theta(gen), init_weight});
		weights.push_back(init_weight);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Process model (curve):
	// x​f​ = x​0​​ +​​ (v/θ_dot)​​ [sin(θ​0​​ +​ θ_dot*dt) − sin(θ​0)] + noise
	// y​f​​ = y​0​​ +​ (v/θ_dot)​​ [cos(θ​0) − cos(θ​0​​ + ​θ_dot*dt)] + noise
	// θ​f​​ = θ​0​​ +​ θ_dot*dt + noise
	// Process model (straight):
	// x​f​ = x​0​​ +​​ v*dt*cos(θ0) + noise
	// y​f​​ = y​0​​ +​ v*dt*sin(θ0) + noise
	// θ​f​​ = 0 + noise

	// Check applicable process model: Straight road or curve
	const bool is_moving_straight = fabs(yaw_rate) < 1e-3; 

	// Prepare random Gaussian noise
	default_random_engine gen;
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);

	// Add measurements to each particle and add random Gaussian noise
	for (auto &particle: particles) {
		// Precompute reusable values here
		const double sin_theta = sin(particle.theta);
		const double cos_theta = cos(particle.theta);
		const double new_theta = particle.theta + yaw_rate * delta_t;

		if (is_moving_straight) {
			// Straight model
			particle.x += velocity * delta_t * cos_theta + noise_x(gen);
			particle.y += velocity * delta_t * sin_theta + noise_y(gen);
			particle.theta += noise_theta(gen);
		} else {
			// Curve model
			particle.x += (velocity / yaw_rate) * (sin(new_theta) - sin_theta) + noise_x(gen);
			particle.y += (velocity / yaw_rate) * (cos_theta - cos(new_theta)) + noise_y(gen);
			particle.theta = new_theta + noise_theta(gen);
		}
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// Used as a helper during during the updateWeights phase.

	//dataAssociation(landmarks_in_sensor_range, observations_map_cs);


	// min_element function used as helper.
	// http://www.cplusplus.com/reference/algorithm/min_element/
	// http://www.geeksforgeeks.org/stdmin_element-in-cpp/
	/*
	   The min_element method here uses a user defined function to compare each 
	     landmark in the sensor range to the current observation in the loop.
	   The user defined function measure the distance to observation for the 2 landmarks
	     being compared at the moment (within in_element method algorithm).
	   So in the end, the min_element method will pick the closest landmark.
	*/ 
	for (auto &observation: observations) {
		auto nearest_landmark = min_element(predicted.begin(),
		                                    predicted.end(),
		                                    [&](const LandmarkObs &mark1, const LandmarkObs &mark2) {
		                                    	// Evaluate distances from the 2 landmarks to the observation
												const double mark1_distance = dist(mark1.x, mark1.y, observation.x, observation.y);
												const double mark2_distance = dist(mark2.x, mark2.y, observation.x, observation.y);

												// Return bool required by min_element method
												// Return bool must indicate whether mark1 is less than mark2
		                                        return mark1_distance < mark2_distance;
		                                    }
		                                    );

		// Record back in the transformed observations (map cs),
		//   using as id the same identification number used for the nearest landmark
		observation.id = nearest_landmark->id;

	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

	// 1. Transform observations from car to map coordinate system, with respect to the particle.
	//    x​m​​ = xp​​ + (cosθ*x​c) − (sinθ*y​c)
	//    y​m​​ = y​p​​ + (sinθ*x​c​​) + (cosθ*y​c)
	//    xp,yp,θ = particle position [m] and orientation [rad] in map coordinates
	//    xc,yc = observation position [m] in car coordinates
	//    xm,ym = observation position [m] in map coordinates, with respect to the particle
	//
	// 2. Identify only the landmarks within sensor range to work with.
	// 3. Use dataAssociation method to identify nearest landmark to each transformed observation.
	// 4. Use multi-Variate Normal distribution to calculate the likelihood of each particle based on each observation.
	//    https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	//    P(x,y)= ​[1/(2π*σ​x*σ​y)] * e​−[​​ (x−μ​x)^​2 / (2*σ​x​^2)​​​​ + ​​(y−μ​y)^​2​ / (2*σ​y^​2)​ ]
	//	  x,y = observations in map coordinates
	//    μ​x,μ​y = coordinates of the nearest landmarks
	//    σ​x,σ​y = landmark measurement uncertainty


	weights.clear(); // Reset the weights

	for (auto &particle: particles) { // Loop over each particle

        /*****************************************************************************************************************
		1. Transform observations from car to map coordinate system, with respect to the particle.
		*****************************************************************************************************************/

		// Precompute reusable values here
		const double sin_theta = sin(particle.theta);
		const double cos_theta = cos(particle.theta);

		// Transform observations and store in a new variable: observations_map_cs
		vector<LandmarkObs> observations_map_cs;
		for (const auto &observation_car_cs : observations) {
			observations_map_cs.push_back(
			LandmarkObs {observation_car_cs.id,
			             observation_car_cs.x * cos_theta - observation_car_cs.y * sin_theta + particle.x,
			             observation_car_cs.x * sin_theta + observation_car_cs.y * cos_theta + particle.y
			});
		}

		/*****************************************************************************************************************
		2. Identify only the landmarks within sensor range to work with.
		*****************************************************************************************************************/

		// Store workable landmarks in a new variable: landmarks_in_sensor_range
		vector<LandmarkObs> landmarks_in_sensor_range;
		for (const auto &landmark: map_landmarks.landmark_list) {

			// Evaluate by distance, from particle to landmark
			const double distance = dist(landmark.x_f, landmark.y_f, particle.x, particle.y); 

			if (distance <= sensor_range) {
				// Store only the landmarks within sensor range
				landmarks_in_sensor_range.push_back(
				LandmarkObs {static_cast<int>(landmarks_in_sensor_range.size()), landmark.x_f, landmark.y_f}
				);	
			}
		}

		/*****************************************************************************************************************
		3. Use dataAssociation method to identify nearest landmark to each transformed observation.
		*****************************************************************************************************************/

		// Get associations ids
		dataAssociation(landmarks_in_sensor_range, observations_map_cs);


		/*****************************************************************************************************************
		4. Use multi-Variate Normal distribution to calculate the likelihood of each particle based on each observation.
		*****************************************************************************************************************/

		// P(x,y)= ​[1/(2π*σ​x*σ​y)] * e​−[​​ (x−μ​x)^​2 / (2*σ​x​^2)​​​​ + ​​(y−μ​y)^​2​ / (2*σ​y^​2)​ ]
		// x,y = observations in map coordinates
		// μ​x,μ​y = coordinates of the nearest landmarks
		// σ​x,σ​y = landmark measurement uncertainty

		// Reset weight probability
		double weight = 1;

		// Precompute shorter names for uncertainty
		const double std_x = std_landmark[0];
		const double std_y = std_landmark[1];

		// Precompute scalers for equation simplification
		const double scaler = 1.0 / (2.0 * M_PI * std_x * std_y);
		const double scaler_x = 2.0 * pow(std_x, 2);
		const double scaler_y = 2.0 * pow(std_y, 2);		

		// Loop over each observation for total probability
		for (const auto &map_observation: observations_map_cs) {
			// Nearest landmark 
			auto nearest_landmark = landmarks_in_sensor_range[map_observation.id];
			// Precompute square differences for equation simplification
			const double dx2 = pow(map_observation.x - nearest_landmark.x, 2);
			const double dy2 = pow(map_observation.y - nearest_landmark.y, 2);
			// Final multi-Variate Normal distribution equation
			weight *= scaler * exp(-(dx2 / scaler_x + dy2 / scaler_y));
			// Optimization
			if (weight == 0)
				break;
		}

		// Finally, update weight and push on record
		particle.weight = weight;
		weights.push_back(weight);

	} //end particle for

}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 

	// Reset initial weight to 1.
	// Create new temporary pool of particles.
	const double init_weight = 1.0;
	vector<Particle> new_particles;


	// std::discrete_distribution was helpful here:
	// http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<int> probable_particle_index_sampler(weights.begin(), weights.end());

	// Draw new particles from the current particle pool, based on their probability weight
	for (int i = 0; i < num_particles; ++i) {
		int index = probable_particle_index_sampler(gen);

		new_particles.push_back(Particle {i,
		                                  particles[index].x,
		                                  particles[index].y,
		                                  particles[index].theta,
		                                  init_weight});

	}

	// Resampled particles
	particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	// Particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// Associations: The landmark id that goes along with each listed association
	// Sense_x: the associations x mapping already converted to world coordinates
	// Sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // Get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // Get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // Get rid of the trailing space
    return s;
}
