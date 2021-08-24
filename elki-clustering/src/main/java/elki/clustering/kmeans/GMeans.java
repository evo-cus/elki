/*
 * This file is part of ELKI:
 * Environment for Developing KDD-Applications Supported by Index-Structures
 * 
 * Copyright (C) 2021
 * ELKI Development Team
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
package elki.clustering.kmeans;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import elki.clustering.kmeans.initialization.KMeansInitialization;
import elki.clustering.kmeans.initialization.Predefined;
import elki.data.Cluster;
import elki.data.Clustering;
import elki.data.NumberVector;
import elki.data.model.MeanModel;
import elki.database.ids.DBIDIter;
import elki.database.relation.ProxyView;
import elki.database.relation.Relation;
import elki.distance.NumberVectorDistance;
import elki.distance.minkowski.SquaredEuclideanDistance;
import elki.logging.Logging;
import elki.logging.progress.MutableProgress;
import elki.logging.statistics.LongStatistic;
import elki.logging.statistics.StringStatistic;
import elki.math.linearalgebra.VMath;
import elki.math.statistics.tests.AndersonDarlingTest;
import elki.result.Metadata;
import elki.utilities.datastructures.iterator.It;
import elki.utilities.documentation.Reference;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.WrongParameterValueException;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.ChainedParameterization;
import elki.utilities.optionhandling.parameterization.ListParameterization;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.DoubleParameter;
import elki.utilities.optionhandling.parameters.IntParameter;
import elki.utilities.optionhandling.parameters.ObjectParameter;
import elki.utilities.optionhandling.parameters.RandomParameter;
import elki.utilities.random.RandomFactory;

/**
 * G-Means extends K-Means and estimates the number of centers with Anderson
 * Darling Test.
 * 
 * <p>
 * Reference:
 * <p>
 * Greg Hamerly and Charles Elkan<br>
 * Learning the K in K-Means<br>
 * Advances in Neural Information Processing Systems 17 (NIPS 2004)
 * 
 * @author Robert Gehde
 *
 * @param <V> Vector
 * @param <M> Model
 */
@Reference(authors = "Greg Hamerly and Charles Elkan", //
    booktitle = "Advances in Neural Information Processing Systems 17 (NIPS 2004)", //
    title = "Learning the k in k-means", //
    url = "https://www.researchgate.net/publication/2869155_Learning_the_K_in_K-Means")
public class GMeans<V extends NumberVector, M extends MeanModel> extends AbstractKMeans<V, M> {
  /**
   * The logger for this class.
   */
  private static final Logging LOG = Logging.getLogger(GMeans.class);

  /**
   * Key for statistics logging.
   */
  private static final String KEY = GMeans.class.getName();

  /**
   * Inner k-means algorithm.
   */
  private KMeans<V, M> innerKMeans;

  /**
   * Effective number of clusters, minimum and maximum.
   */
  private int k, k_min, k_max;

  /**
   * Initializer for k-means.
   */
  Predefined splitInitializer;

  /**
   * Random factory.
   */
  RandomFactory rnd;

  /**
   * Random object.
   */
  Random rand;

  /**
   * Significance level.
   */
  double alpha;

  public GMeans(NumberVectorDistance<? super V> distance, double alpha, int k_min, int k_max, int maxiter, KMeans<V, M> innerKMeans, KMeansInitialization initializer, RandomFactory random) {
    super(distance, k_min, maxiter, initializer);
    this.alpha = alpha;
    this.k_min = k_min;
    this.k_max = k_max;
    this.k = k_min;
    this.innerKMeans = innerKMeans;
    this.splitInitializer = new Predefined((double[][]) null);
    this.innerKMeans.setInitializer(this.splitInitializer);
    this.innerKMeans.setDistance(distance);
    this.rnd = random;
    this.rand = this.rnd.getRandom();
  }

  @Override
  public Clustering<M> run(Relation<V> relation) {
    MutableProgress prog = LOG.isVerbose() ? new MutableProgress("G-means number of clusters", k_max, LOG) : null;

    // Run initial k-means to find at least k_min clusters
    innerKMeans.setK(k_min);
    LOG.statistics(new StringStatistic(KEY + ".initialization", initializer.toString()));
    splitInitializer.setInitialMeans(initializer.chooseInitialMeans(relation, k_min, distance));
    Clustering<M> clustering = innerKMeans.run(relation);

    if(prog != null) {
      prog.setProcessed(k_min, LOG);
    }

    ArrayList<Cluster<M>> clusters = new ArrayList<>(clustering.getAllClusters());
    while(clusters.size() <= k_max) {
      // Improve-Structure:
      ArrayList<Cluster<M>> nextClusters = new ArrayList<>();
      for(Cluster<M> cluster : clusters) {
        // Try to split this cluster:
        List<Cluster<M>> childClusterList = splitCluster(cluster, relation);
        nextClusters.addAll(childClusterList);
        if(childClusterList.size() > 1) {
          k += childClusterList.size() - 1;
          if(prog != null) {
            if(k >= k_max) {
              prog.setTotal(k + 1);
            }
            prog.setProcessed(k, LOG);
          }
        }
      }
      if(clusters.size() == nextClusters.size()) {
        break;
      }
      // Improve-Params:
      splitInitializer.setInitialClusters(nextClusters);
      innerKMeans.setK(nextClusters.size());
      innerKMeans.setInitializer(splitInitializer);
      clustering = innerKMeans.run(relation);
      clusters.clear();
      clusters.addAll(clustering.getAllClusters());
    }

    // Ensure that the progress bar finished.
    if(prog != null) {
      prog.setTotal(k);
      prog.setProcessed(k, LOG);
    }
    LOG.statistics(new LongStatistic(KEY + ".num-clusters", clusters.size()));
    Clustering<M> result = new Clustering<>(clusters);
    Metadata.of(result).setLongName("G-Means Clustering");
    return result;
  }

  /**
   * Conditionally splits the clusters based on the information criterion.
   *
   * @param parentCluster Cluster to split
   * @param relation Data relation
   * @return Parent cluster when split decreases clustering quality or child
   *         clusters when split improves clustering.
   */
  protected List<Cluster<M>> splitCluster(Cluster<M> parentCluster, Relation<V> relation) {
    // Transform parent cluster into a clustering
    ArrayList<Cluster<M>> parentClusterList = new ArrayList<>(1);
    parentClusterList.add(parentCluster);
    if(parentCluster.size() <= 1) {
      // Split is not possbile
      return parentClusterList;
    }
    // splitting
    ProxyView<V> parentview = new ProxyView<V>(parentCluster.getIDs(), relation);
    int dim = relation.get(relation.iterDBIDs()).getDimensionality();
    int n = parentCluster.size();
    // calculate new centers
    // 0: get points vectors
    double[][] points = new double[n][];
    int c = 0;
    for(DBIDIter it = parentview.iterDBIDs(); it.valid(); it.advance()) {
      points[c++] = relation.get(it).toArray();
    }
    // 1: calc old center c
    double[] center = new double[dim];
    for(int i = 0; i < points.length; i++) {
      for(int j = 0; j < center.length; j++) {
        center[j] += points[i][j];
      }
    }
    for(int j = 0; j < center.length; j++) {
      center[j] /= n;
    }
    // 2: calculate eigenvector
    // 2.1: calc cov
    for(int i = 0; i < points.length; i++) {
      points[i] = VMath.minusEquals(points[i], center);
    }
    double[][] cov = VMath.timesEquals(VMath.transposeTimes(points, points), 1.0 / (n - (1.0)));
    // 2.2: main principal component via power method
    double[] s = new double[dim];
    for(int i = 0; i < s.length; i++) {
      s[i] = rand.nextDouble();
    }
    VMath.normalize(s);
    for(int i = 0; i < 100; i++) {
      s = VMath.times(cov, s);
      s = VMath.normalize(s);
    }
    // 2.3: Eigenvalue
    double l = VMath.transposeTimesTimes(s, cov, s);
    // 3: deviation is m = s * sqrt(2l/pi)
    double[] m = VMath.times(s, Math.sqrt(2 * l / Math.PI));
    // 4: new centers are c +/- m
    double[][] newCenters = new double[2][dim];
    newCenters[0] = VMath.plus(center, m);
    newCenters[1] = VMath.minus(center, m);
    Predefined init = new Predefined(newCenters);

    // run it a bit
    innerKMeans.setK(2);
    innerKMeans.setInitializer(init);
    // ich würde das gerne nur 1mal laufen lassen....
    Clustering<M> childClustering = innerKMeans.run(parentview);
    c = 0;
    for(It<Cluster<M>> it = childClustering.iterToplevelClusters(); it.valid(); it.advance()) {
      newCenters[c++] = it.get().getModel().getMean();
    }
    // evaluation
    // v = c2 - c1 = 2m
    double[] v = VMath.minus(newCenters[1], newCenters[0]);
    double length = VMath.euclideanLength(v);
    double[] projectedValues = new double[n];
    for(int i = 0; i < projectedValues.length; i++) {
      projectedValues[i] = VMath.dot(points[i], v) / length;
    }
    // transform data to center 0 and var 1
    normalize(projectedValues, n);
    // test
    Arrays.sort(projectedValues);
    double A2 = AndersonDarlingTest.A2StandardNormal(projectedValues);
    A2 = AndersonDarlingTest.removeBiasNormalDistribution(A2, n);
    if(LOG.isDebugging()) {
      LOG.debug("AndersonDarlingValue: " + A2);
    }
    // Check if split is an improvement:
    return pValueAdjA2(A2) > alpha ? parentClusterList : childClustering.getAllClusters();
  }

  /**
   * normalizes the values such that mean is 0 and variance is 1
   */
  private void normalize(double[] data, int n) {
    double mean = 0;
    for(int i = 0; i < data.length; i++) {
      mean += data[i];
    }
    mean /= n;
    double sig = 0;
    for(int i = 0; i < data.length; i++) {
      sig += Math.pow(data[i] - mean, 2);
    }
    sig = Math.sqrt(1.0 / (n - 1.0) * sig);
    for(int i = 0; i < data.length; i++) {
      data[i] = (data[i] - mean) / sig;
    }
  }

  /**
   * calculate p-value for adjusted Anderson Darling test and case 3
   * 
   * @param A2
   * @return
   */
  private double pValueAdjA2(double A2) {
    if(A2 >= 0.6) {
      return Math.exp(1.2937 - 5.709 * A2 + 0.0186 * A2 * A2);
    }
    else if(A2 >= 0.34) {
      return Math.exp(0.9177 - 4.279 * A2 - 1.38 * A2 * A2);
    }
    else if(A2 >= 0.2) {
      return 1 - Math.exp(-8.318 + 42.796 * A2 - 59.938 * A2 * A2);
    }
    else {
      return 1 - Math.exp(-13.436 - 101.14 * A2 + 223.73 * A2 * A2);
    }
  }

  @Override
  protected Logging getLogger() {
    return LOG;
  }

  /**
   * Parameterization class.
   *
   * @author Tibor Goldschwendt
   * @author Erich Schubert
   * @author Robert Gehde
   *
   * @hidden
   *
   * @param <V> Vector type
   * @param <M> Model type of inner algorithm
   */
  public static class Par<V extends NumberVector, M extends MeanModel> extends AbstractKMeans.Par<V> {
    /**
     * Parameter to specify the kMeans variant.
     */
    public static final OptionID INNER_KMEANS_ID = new OptionID("gmeans.kmeans", "kMeans algorithm to use.");

    /**
     * Minimum number of clusters.
     */
    public static final OptionID K_MIN_ID = new OptionID("gmeans.k_min", "The minimum number of clusters to find.");

    /**
     * Randomization seed.
     */
    public static final OptionID SEED_ID = new OptionID("gmeans.seed", "Random seed for splitting clusters.");

    /**
     * Significance level.
     */
    public static final OptionID ALPHA_ID = new OptionID("gmeans.alpha", "Significance level for the Anderson Darling test.");

    /**
     * Variant of kMeans
     */
    protected KMeans<V, M> innerKMeans;

    /**
     * Minimum and maximum number of result clusters.
     */
    protected int k_min, k_max;

    /**
     * Significance level.
     */
    protected double alpha;

    /**
     * Random number generator.
     */
    private RandomFactory random;

    @Override
    public void configure(Parameterization config) {
      // Do NOT invoke super.makeOptions to hide the "k" parameter.
      IntParameter kMinP = new IntParameter(K_MIN_ID, 2) //
          .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT);
      kMinP.grab(config, x -> k_min = x);
      DoubleParameter alphaP = new DoubleParameter(ALPHA_ID, 0.0001) //
          .addConstraint(CommonConstraints.GREATER_THAN_ZERO_DOUBLE) //
          .addConstraint(CommonConstraints.LESS_EQUAL_ONE_DOUBLE);
      alphaP.grab(config, x -> alpha = x);

      IntParameter kMaxP = new IntParameter(KMeans.K_ID) //
          .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT);
      kMaxP.grab(config, x -> k_max = x);
      // Non-formalized parameter constraint: k_min <= k_max
      if(k_min > k_max) {
        config.reportError(new WrongParameterValueException(kMinP, "must be at most", kMaxP, ""));
      }

      getParameterInitialization(config);
      getParameterMaxIter(config);
      getParameterDistance(config);

      new RandomParameter(SEED_ID).grab(config, x -> random = x);
      ObjectParameter<KMeans<V, M>> innerKMeansP = new ObjectParameter<>(INNER_KMEANS_ID, KMeans.class, LloydKMeans.class);
      if(config.grab(innerKMeansP)) {
        ChainedParameterization combinedConfig = new ChainedParameterization(new ListParameterization() //
            .addParameter(KMeans.K_ID, k_min) //
            .addParameter(KMeans.INIT_ID, new Predefined((double[][]) null)) //
            .addParameter(KMeans.MAXITER_ID, maxiter) //
            // Setting the distance to null if undefined at this point will
            // cause validation errors later. So fall back to the default.
            .addParameter(KMeans.DISTANCE_FUNCTION_ID, distance != null ? //
                distance : SquaredEuclideanDistance.STATIC), config);
        combinedConfig.errorsTo(config);
        innerKMeans = innerKMeansP.instantiateClass(combinedConfig);
      }

    }

    @Override
    public GMeans<V, M> make() {
      return new GMeans<>(distance, alpha, k_min, k_max, maxiter, innerKMeans, initializer, random);
    }
  }
}
