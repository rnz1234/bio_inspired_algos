import csv
from config import *



# reader class
class Reader(object):
    # constructor
    def __init__(self, training_set_path, validation_set_path, testing_set_path):
        self.training_set_path = training_set_path
        self.validation_set_path = validation_set_path
        self.testing_set_path = testing_set_path

    # reading a CSV samples file.
    # output format:
    # - np.array where each index (row) is an np.array (2d array), index 0 in each row is the label
    # - if is_test_set = True, index 0 in each row is assigned -1 (marking no label)
    def _read_file(self, file_path, inc_header=False, is_test_set=False):
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            samples = []

            line = 0
            for row in csv_reader:
                if inc_header:
                    if line == 0:
                        line += 1
                        continue
                if is_test_set:
                    row[0] = -1

                samples.append(np.array(row[:]).astype(np.float))

                line += 1
                # limit number of read lines for debug
                if DEBUG_LIMIT_LINES_READ:
                    if line >= DEBUG_NUM_OF_LINES_READ_LIMIT:
                        break

                if line % 1000 == 0:
                    print("Reader : line number = " + str(line))

            samples = np.vstack(samples)

            return samples

    # reading training set
    def read_training_set(self):
        print("Reader : reading training set")
        self.training_set = self._read_file(file_path=self.training_set_path)
        if DEBUG_READER:
            print(self.training_set)

    # reading validation set
    def read_validation_set(self):
        print("Reader : reading validation set")
        self.validation_set = self._read_file(file_path=self.validation_set_path)
        if DEBUG_READER:
            print(self.validation_set)

    # reading test set
    def read_testing_set(self):
        print("Reader : reading testing set")
        self.testing_set = self._read_file(file_path=self.testing_set_path, is_test_set=True)
        if DEBUG_READER:
            print(self.testing_set)

    # get method for training set
    def get_training_set(self):
        return self.training_set

    # get method for validation set
    def get_validation_set(self):
        return self.validation_set

    # get method for testing set
    def get_testing_set(self):
        return self.testing_set


# Writer class
class Writer(object):
    # constructor
    def __init__(self, output_path):
        self.output_path = output_path

    # write int data
    def write(self, data):
        np.savetxt(fname=self.output_path, X=data, fmt='%d')

    # write float data
    def write_float(self, data):
        np.savetxt(fname=self.output_path, X=data, fmt='%f')

# unit test
if __name__ == '__main__':
    r = Reader(training_set_path="train.csv",
               validation_set_path="validate.csv",
               testing_set_path="test.csv")

    r.read_training_set()
    r.read_validation_set()
    r.read_testing_set()

    print("training_set : " + str(r.get_training_set()))
    print("validation_set : " + str(r.get_validation_set()))
    print("testing_set : " + str(r.get_testing_set()))

    print("training labels: " + str(r.get_training_set()[:,0]))
    print("training features: " + str(r.get_training_set()[:, 1:]))