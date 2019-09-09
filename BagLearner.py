import numpy as np

class BagLearner(object):

    def __init__(self, learner, kwargs={"leaf_size":1}, bags=20, boost=False, verbose=False):
        #pass  # move along, these aren't the drones you're looking for
        self.learner=learner
        self.learners = []       # array of learners

        # Instantiate learners with kwargs{} parameters and load into the learners array
        for i in range(0, bags):
            self.learners.append(learner(**kwargs))

        self.bags = bags

    def author(self):
        return 'mmiller319'  # replace tb34 with your Georgia Tech username

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        # Create a list of indexes numbering 0 to the number of row in dataX
        #dataX_numRows = dataX.shape[0]
        indexes = np.arange(0, dataX.shape[0]-1)

        # Each bag needs to be trained on a different subset of the data, so for each learner in
        # self.learners, a random index is chosen in order to pass a different subset to the
        # addEvidence() method for that learner.
        for currentLearner in self.learners:
            index = np.random.choice(indexes, indexes.size)
            currentDataX = dataX.take(index, axis=0)
            currentDataY = dataY.take(index, axis=0)

            # index = np.random.randint(0, indexes.size-1)
            #currentDataX = dataX[0, index]
            #currentDataY = dataY[0, index]

            currentLearner.addEvidence(currentDataX, currentDataY)


    # Query each learner and return mean of query results
    def query(self, points):

        queryResults = []

        for currentLearner in self.learners:
            queryResults.append(currentLearner.query(points))

        queryResults = np.array(queryResults)

        return np.mean(queryResults, axis=0)



if __name__ == "__main__":
    print ""
