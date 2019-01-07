/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
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
  
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 100;
  
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];
  
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);
  
  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    
    particles.push_back(p);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  double std_pos_x = std_pos[0];
  double std_pos_y = std_pos[1];
  double std_pos_theta = std_pos[2];

  normal_distribution<double> dist_x(0, std_pos_x);
  normal_distribution<double> dist_y(0, std_pos_y);
  normal_distribution<double> dist_theta(0, std_pos_theta);
  
  for (int i = 0; i < num_particles; i++) {
    Particle p = particles[i];

    double new_x;
    double new_y;
    double new_theta;

    if (fabs(yaw_rate) < 0.0001) {
      new_x = p.x + velocity * delta_t * cos(p.theta);
      new_y = p.y + velocity * delta_t * sin(p.theta);
      new_theta = p.theta;
    } else {
      double coefficient = velocity / yaw_rate;
      double parameter = p.theta + (yaw_rate * delta_t);

      new_x = p.x + coefficient * (sin(parameter) - sin(p.theta));
      new_y = p.y + coefficient * (cos(p.theta) - cos(parameter));
      new_theta = parameter;
    }
  
    // add noise
    particles[i].x = new_x + dist_x(gen);
    particles[i].y = new_y + dist_y(gen);
    particles[i].theta = new_theta + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.

  for (int i = 0; i < observations.size(); i++) {
    double minDist = numeric_limits<double>::max();
    int minId = -1;
    LandmarkObs o = observations[i];
    for (int j = 0; j < predicted.size(); j++) {
      LandmarkObs p = predicted[i];
      // utilize helper function, 3rd param x2 == x_obs
      double distVal = dist(p.x, p.y, o.x, o.y);
      if (distVal < minDist) {
        minDist = distVal;
        minId = p.id;
      }
    }

    // set the min id to observation
    observations[i].id = minId;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
    const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // TODO: Update the weights of each particle using a multi-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];

  double gauss_norm = (1.0 / (2.0 * M_PI * sig_x * sig_y));
  
  for (int i = 0; i < num_particles; i++) {
    Particle p = particles[i];

    // filter only landmarks within sensor_range
    std::vector<LandmarkObs> landmark;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;
      
      if (dist(lm_x, lm_y, p.x, p.y) <= sensor_range) {
        LandmarkObs pred;
        pred.x = lm_x;
        pred.y = lm_y;
        pred.id = lm_id;

        landmark.push_back(pred);
      }
    }

    std::vector<LandmarkObs> map_observations;
    // convert observations to map coordinate system
    for (int k = 0; k < observations.size(); k++) {
      LandmarkObs c = observations[k];
      
      // homogeneous transformation (rotation + translation)
      LandmarkObs map_obs;
      map_obs.x = c.x * cos(p.theta) - c.y * sin(p.theta) + p.x;
      map_obs.y = c.x * sin(p.theta) + c.y * cos(p.theta) + p.y;
      map_obs.id = c.id;
      
      map_observations.push_back(map_obs);
    }

    // update observations with the nearest neighbor id after particle transformation
    dataAssociation(landmark, map_observations);

    // update particle weights
    particles[i].weight = 1.0;

    for (int x = 0; x < map_observations.size(); x++) {
      LandmarkObs m = map_observations[x];

      double x_obs = m.x;
      double y_obs = m.y;
      double mu_x;
      double mu_y;

      for (int y = 0; y < landmark.size(); y++) {
        LandmarkObs mu = landmark[y];

        if (mu.id == m.id) {
          mu_x = mu.x;
          mu_y = mu.y;
        }
      }

      double exponent = pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)) +
                        pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2));

      double weight = gauss_norm * exp(-exponent);
      if (weight < 0.0001) {
        weight = 0.0001;
      }
      particles[i].weight *= weight;
    }
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // get max weight
  weights.clear();
  double maxWeight = 0.0;
  for (int i = 0; i < num_particles; i++) {
    double weight = particles[i].weight;
    
    weights.push_back(weight);
    if (weight > maxWeight) {
      maxWeight = weight;
    }
  }
  uniform_int_distribution<int> distIndex(0, num_particles - 1);
  int index = distIndex(gen);

  uniform_real_distribution<double> distBeta(0.0, 2.0 * maxWeight);
  double beta = 0.0;

  vector<Particle> resample_particles;
  for (int j = 0; j < num_particles; j++) {
    beta += distBeta(gen);

    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }

    resample_particles.push_back(particles[index]);
  }

  particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
