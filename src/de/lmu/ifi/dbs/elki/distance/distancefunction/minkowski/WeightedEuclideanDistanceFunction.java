package de.lmu.ifi.dbs.elki.distance.distancefunction.minkowski;

/*
 This file is part of ELKI:
 Environment for Developing KDD-Applications Supported by Index-Structures

 Copyright (C) 2013
 Ludwig-Maximilians-Universität München
 Lehr- und Forschungseinheit für Datenbanksysteme
 ELKI Development Team

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

import java.util.Arrays;

import de.lmu.ifi.dbs.elki.data.NumberVector;
import de.lmu.ifi.dbs.elki.data.spatial.SpatialComparable;

/**
 * Provides the Euclidean distance for FeatureVectors.
 * 
 * @author Erich Schubert
 */
public class WeightedEuclideanDistanceFunction extends WeightedLPNormDistanceFunction {
  /**
   * Constructor.
   * 
   * @param weights
   */
  public WeightedEuclideanDistanceFunction(double[] weights) {
    super(2.0, weights);
  }

  @Override
  protected double doublePreDistance(NumberVector<?> v1, NumberVector<?> v2, final int start, final int end, double agg) {
    for (int d = start; d < end; d++) {
      final double xd = v1.doubleValue(d), yd = v2.doubleValue(d);
      final double delta = xd - yd;
      agg += delta * delta * weights[d];
    }
    return agg;
  }

  @Override
  protected double doublePreDistanceVM(NumberVector<?> v, SpatialComparable mbr, final int start, final int end, double agg) {
    for (int d = start; d < end; d++) {
      final double value = v.doubleValue(d), min = mbr.getMin(d);
      double delta = min - value;
      if (delta < 0.) {
        delta = value - mbr.getMax(d);
      }
      if (delta > 0.) {
        agg += delta * delta * weights[d];
      }
    }
    return agg;
  }

  @Override
  protected double doublePreDistanceMBR(SpatialComparable mbr1, SpatialComparable mbr2, final int start, final int end, double agg) {
    for (int d = start; d < end; d++) {
      double delta = mbr2.getMin(d) - mbr1.getMax(d);
      if (delta < 0.) {
        delta = mbr1.getMin(d) - mbr2.getMax(d);
      }
      if (delta > 0.) {
        agg += delta * delta * weights[d];
      }
    }
    return agg;
  }

  @Override
  protected double doublePreNorm(NumberVector<?> v, final int start, final int end, double agg) {
    for (int d = start; d < end; d++) {
      final double xd = v.doubleValue(d);
      agg += xd * xd * weights[d];
    }
    return agg;
  }

  @Override
  protected double doublePreNormMBR(SpatialComparable mbr, final int start, final int end, double agg) {
    for (int d = start; d < end; d++) {
      double delta = mbr.getMin(d);
      if (delta < 0.) {
        delta = -mbr.getMax(d);
      }
      if (delta > 0.) {
        agg += delta * delta * weights[d];
      }
    }
    return agg;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (!(obj instanceof WeightedEuclideanDistanceFunction)) {
      if (obj.getClass().equals(WeightedLPNormDistanceFunction.class)) {
        return super.equals(obj);
      }
      if (obj.getClass().equals(EuclideanDistanceFunction.class)) {
        for (double d : weights) {
          if (d != 1.0) {
            return false;
          }
        }
        return true;
      }
      return false;
    }
    WeightedEuclideanDistanceFunction other = (WeightedEuclideanDistanceFunction) obj;
    return Arrays.equals(this.weights, other.weights);
  }
}
