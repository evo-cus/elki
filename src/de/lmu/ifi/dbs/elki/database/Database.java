package de.lmu.ifi.dbs.elki.database;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import de.lmu.ifi.dbs.elki.data.ClassLabel;
import de.lmu.ifi.dbs.elki.data.DatabaseObject;
import de.lmu.ifi.dbs.elki.database.datastore.DataStoreEvent;
import de.lmu.ifi.dbs.elki.database.datastore.DataStoreListener;
import de.lmu.ifi.dbs.elki.database.ids.DBID;
import de.lmu.ifi.dbs.elki.database.ids.DBIDs;
import de.lmu.ifi.dbs.elki.database.query.distance.DistanceQuery;
import de.lmu.ifi.dbs.elki.database.query.knn.KNNQuery;
import de.lmu.ifi.dbs.elki.database.query.range.RangeQuery;
import de.lmu.ifi.dbs.elki.database.query.rknn.RKNNQuery;
import de.lmu.ifi.dbs.elki.distance.distancefunction.DistanceFunction;
import de.lmu.ifi.dbs.elki.distance.distancevalue.Distance;
import de.lmu.ifi.dbs.elki.logging.Logging;
import de.lmu.ifi.dbs.elki.result.Result;
import de.lmu.ifi.dbs.elki.utilities.exceptions.UnableToComplyException;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.OptionID;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.Parameterizable;
import de.lmu.ifi.dbs.elki.utilities.pairs.Pair;

/**
 * Database specifies the requirements for any database implementation. Note
 * that any implementing class is supposed to provide a constructor without
 * parameters for dynamic instantiation.
 * 
 * @author Elke Achtert
 * @param <O> the type of DatabaseObject as element of the database
 */
public interface Database<O extends DatabaseObject> extends Result, Iterable<DBID>, Parameterizable {
  /**
   * Inserts the given objects and their associations into the database.
   * 
   * @param objectsAndAssociationsList the list of objects and their
   *        associations to be inserted
   * @return the IDs assigned to the inserted objects
   * @throws UnableToComplyException if insertion is not possible
   */
  DBIDs insert(List<Pair<O, DatabaseObjectMetadata>> objectsAndAssociationsList) throws UnableToComplyException;

  /**
   * Inserts the given object and its associations into the database.
   * 
   * @param objectAndAssociations the object and its associations to be inserted
   * @return the ID assigned to the inserted object
   * @throws UnableToComplyException if insertion is not possible
   */
  DBID insert(Pair<O, DatabaseObjectMetadata> objectAndAssociations) throws UnableToComplyException;

  /**
   * Removes and returns the object with the given id from the database.
   * 
   * @param id the id of an object to be removed from the database
   * @return the object that has been removed
   * @throws UnableToComplyException if deletion is not possible
   */
  O delete(DBID id);

  /**
   * Returns the number of objects contained in this Database.
   * 
   * @return the number of objects in this Database
   */
  int size();

  /**
   * Get the object factory for this data type.
   * 
   * @return object factory
   */
  O getObjectFactory();

  /**
   * Set the object factory.
   * 
   * @param objectFactory Object factory
   */
  void setObjectFactory(O objectFactory);

  /**
   * Returns a random sample of k ids.
   * 
   * @param k the number of ids to return
   * @param seed for random generator
   * @return a list of k ids
   */
  DBIDs randomSample(int k, long seed);

  /**
   * Get the distance query for a particular distance function.
   * 
   * @param <D> Distance result type
   * @param distanceFunction Distance function to use
   * @return Instance to query the database with this distance
   */
  <D extends Distance<D>> DistanceQuery<O, D> getDistanceQuery(DistanceFunction<? super O, D> distanceFunction);

  /**
   * Get a KNN query object for the given distance function.
   * 
   * When possible, this will use an index, but it may default to an expensive linear scan.
   * 
   * Hints include:
   * <ul>
   * <li>Integer: maximum value for k needed</li>
   * <li>{@link de.lmu.ifi.dbs.elki.database.query.DatabaseQuery#HINT_BULK} bulk query needed</li>
   * </ul>
   *  
   * @param <D> Distance type
   * @param distanceFunction Distance function
   * @param hints Optimizer hints
   * @return KNN Query object
   */
  <D extends Distance<D>> KNNQuery<O, D> getKNNQuery(DistanceFunction<? super O, D> distanceFunction, Object... hints);

  /**
   * Get a KNN query object for the given distance query.
   * 
   * When possible, this will use an index, but it may default to an expensive linear scan.
   *  
   * Hints include:
   * <ul>
   * <li>Integer: maximum value for k needed</li>
   * <li>{@link de.lmu.ifi.dbs.elki.database.query.DatabaseQuery#HINT_BULK} bulk query needed</li>
   * </ul>
   * 
   * @param <D> Distance type
   * @param distanceQuery Distance query
   * @param hints Optimizer hints
   * @return KNN Query object
   */
  <D extends Distance<D>> KNNQuery<O, D> getKNNQuery(DistanceQuery<O, D> distanceQuery, Object... hints);

  /**
   * Get a range query object for the given distance function.
   * 
   * When possible, this will use an index, but it may default to an expensive linear scan.
   * 
   * Hints include:
   * <ul>
   * <li>Distance object: Maximum query range</li>
   * <li>{@link de.lmu.ifi.dbs.elki.database.query.DatabaseQuery#HINT_BULK} bulk query needed</li>
   * </ul>
   *  
   * @param <D> Distance type
   * @param distanceFunction Distance function
   * @param hints Optimizer hints
   * @return KNN Query object
   */
  <D extends Distance<D>> RangeQuery<O, D> getRangeQuery(DistanceFunction<? super O, D> distanceFunction, Object... hints);

  /**
   * Get a range query object for the given distance query.
   * 
   * When possible, this will use an index, but it may default to an expensive linear scan.
   *  
   * Hints include:
   * <ul>
   * <li>Distance object: Maximum query range</li>
   * <li>{@link de.lmu.ifi.dbs.elki.database.query.DatabaseQuery#HINT_BULK} bulk query needed</li>
   * </ul>
   * 
   * @param <D> Distance type
   * @param distanceQuery Distance query
   * @param hints Optimizer hints
   * @return KNN Query object
   */
  <D extends Distance<D>> RangeQuery<O, D> getRangeQuery(DistanceQuery<O, D> distanceQuery, Object... hints);

  /**
   * Get a rKNN query object for the given distance function.
   * 
   * When possible, this will use an index, but it may default to an expensive linear scan.
   * 
   * Hints include:
   * <ul>
   * <li>Integer: maximum value for k needed</li>
   * <li>{@link de.lmu.ifi.dbs.elki.database.query.DatabaseQuery#HINT_BULK} bulk query needed</li>
   * </ul>
   *  
   * @param <D> Distance type
   * @param distanceFunction Distance function
   * @param hints Optimizer hints
   * @return KNN Query object
   */
  <D extends Distance<D>> RKNNQuery<O, D> getRKNNQuery(DistanceFunction<? super O, D> distanceFunction, Object... hints);

  /**
   * Get a rKNN query object for the given distance query.
   * 
   * When possible, this will use an index, but it may default to an expensive linear scan.
   *  
   * Hints include:
   * <ul>
   * <li>Integer: maximum value for k needed</li>
   * <li>{@link de.lmu.ifi.dbs.elki.database.query.DatabaseQuery#HINT_BULK} bulk query needed</li>
   * </ul>
   * 
   * @param <D> Distance type
   * @param distanceQuery Distance query
   * @param hints Optimizer hints
   * @return KNN Query object
   */
  <D extends Distance<D>> RKNNQuery<O, D> getRKNNQuery(DistanceQuery<O, D> distanceQuery, Object... hints);

  /**
   * Returns the DatabaseObject represented by the specified id.
   * 
   * @param id the id of the Object to be obtained from the Database
   * @return Object the Object represented by to the specified id in the
   *         Database
   */
  O get(DBID id);

  /**
   * Get the object label
   * 
   * (Temporary function for DB layer migration)
   * 
   * @param id Object id
   * @return Label or {@code null}
   */
  String getObjectLabel(DBID id);

  /**
   * Set the object label
   * 
   * (Temporary function for DB layer migration)
   * 
   * @param id Object id
   * @param label new object label
   */
  void setObjectLabel(DBID id, String label);

  /**
   * Get the class label
   * 
   * (Temporary function for DB layer migration)
   * 
   * @param id Object id
   * @return Label or {@code null}
   */
  ClassLabel getClassLabel(DBID id);

  /**
   * Set the class label
   * 
   * (Temporary function for DB layer migration)
   * 
   * @param id Object id
   * @param label new class label
   */
  void setClassLabel(DBID id, ClassLabel label);

  /**
   * Get the external id
   * 
   * (Temporary function for DB layer migration)
   * 
   * @param id Object id
   * @return Label or {@code null}
   */
  String getExternalID(DBID id);

  /**
   * Set the external id
   * 
   * (Temporary function for DB layer migration)
   * 
   * @param id Object id
   * @param externalid new external id
   */
  void setExternalID(DBID id, String externalid);

  /**
   * Returns an iterator iterating over all keys of the database.
   * 
   * 
   * @return an iterator iterating over all keys of the database
   * @see Iterable#iterator() - for a Database {@code db}, this allows the
   *      construct {@code for(Integer id : db) // work with database ids }.
   */
  @Override
  Iterator<DBID> iterator();

  /**
   * Returns a list comprising all IDs currently in use.
   * 
   * The list returned shall not be linked to any actual list possibly hold in
   * the database implementation.
   * 
   * @return a list comprising all IDs currently in use
   */
  DBIDs getIDs();

  /**
   * Returns a Map of partition IDs to Databases of the specified class
   * according to the specified Map of partition IDs to Lists of IDs.
   * 
   * @param partitions a Map of partition IDs to Lists of IDs defining a
   *        partition of the database
   * @param dbClass the class of the databases to be returned, if this argument
   *        is <code>null</code> the returned databases have the same class as
   *        this database
   * @param dbParameters the parameter array of the returned database class,
   *        only necessary if parameter <code>dbClass</code> is not null
   * @return a Map of partition IDs to Databases of the specified class
   *         according to the specified Map of Lists of IDs - the databases in
   *         this map may contain the same objects, but the managing IDs are
   *         generally independent from the IDs in the original database
   * @throws UnableToComplyException in case of problems during insertion or
   *         class instantiation
   */
  Map<Integer, Database<O>> partition(Map<Integer, ? extends DBIDs> partitions, Class<? extends Database<O>> dbClass, Collection<Pair<OptionID, Object>> dbParameters) throws UnableToComplyException;

  /**
   * Returns a Map of partition IDs to Databases according to the specified Map
   * of partition IDs to Lists of IDs. Returns the same result as
   * <code>partition(partitions, null, null)</code>.
   * 
   * @param partitions a Map of partition IDs to Lists of IDs defining a
   *        partition of the database
   * @return a Map of partition IDs to Databases of the specified class
   *         according to the specified Map of Lists of IDs - the databases in
   *         this map may contain the same objects, but the managing IDs are
   *         generally independent from the IDs in the original database
   * @throws UnableToComplyException in case of problems during insertion
   */
  Map<Integer, Database<O>> partition(Map<Integer, ? extends DBIDs> partitions) throws UnableToComplyException;

  /**
   * Returns a partition of this database according to the specified Lists of
   * IDs. The returned database has the same class as this database.
   * 
   * @param ids a Lists of IDs defining a partition of the database
   * @return a partition of this database according to the specified Lists of
   *         IDs - the database may contain the same objects, but the managing
   *         IDs are generally independent from the IDs in the original database
   * @throws UnableToComplyException in case of problems during insertion
   */
  Database<O> partition(DBIDs ids) throws UnableToComplyException;

  /**
   * Returns the dimensionality of the data contained by this database in case
   * of {@link Database O} extends {@link de.lmu.ifi.dbs.elki.data.NumberVector
   * FeatureVector}.
   * 
   * @return the dimensionality of the data contained by this database in case
   *         of O extends FeatureVector
   * @throws UnsupportedOperationException if {@link Database O} does not extend
   *         {@link de.lmu.ifi.dbs.elki.data.NumberVector FeatureVector} or the
   *         database is empty
   */
  public int dimensionality() throws UnsupportedOperationException;

  /**
   * Report page accesses to a logger (when "verbose")
   * 
   * @param logger Logger to report to
   */
  public void reportPageAccesses(Logging logger);

  /**
   * Adds a listener for the <code>DataStoreEvent</code> posted after the
   * content of the database changes.
   * 
   * @param l the listener to add
   * @see #removeDataStoreListener(DataStoreListener)
   * @see DataStoreListener
   * @see DataStoreEvent
   */
  void addDataStoreListener(DataStoreListener<O> l);

  /**
   * Removes a listener previously added with
   * {@link #addDataStoreListener(DataStoreListener)}.
   * 
   * @param l the listener to remove
   * @see #addDataStoreListener(DataStoreListener)
   * @see DataStoreListener
   * @see DataStoreEvent
   */
  void removeDataStoreListener(DataStoreListener<O> l);

  /**
   * Collects all insertion, deletion and update events until
   * {@link #flushDataStoreEvents()} is called.
   * 
   * @see DataStoreEvent
   */
  void accumulateDataStoreEvents();

  /**
   * Fires all collected insertion, deletion and update events as one
   * DataStoreEvent, i.e. notifies all registered DataStoreListener how the
   * content of the database has been changed since
   * {@link #accumulateDataStoreEvents()} has been called.
   * 
   * @see DataStoreListener
   * @see DataStoreEvent
   */
  void flushDataStoreEvents();
  // TODO remaining methods

  // int getNumKNNQueries();

  // void resetNumKNNQueries();

  // int getNumRNNQueries();

  // void resetNumRNNQueries();

  // int getIOAccess();

  // void resetIOAccess();
}