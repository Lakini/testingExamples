package org.deeplearning4j.examples.feedforward.classification;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.Collections;
import java.util.List;

public class MLPClassifierLinearIrisData {

    public static void main(String[] args) throws Exception {
        int seed = 123;
        double learningRate = 0.01;
       // int batchSize = 50;
        int nEpochs = 1;
        //20 iterations

        int numInputs = 4;
        int numOutputs = 3;
        int numHiddenNodes = 20;

        //Load the training data:
        //RecordReader rr = new CSVRecordReader();
       // rr.initialize(new FileSplit(new File("C:/Users/Lakini/Desktop/CSVData/iris_data_training.csv")));
        //4 is the index of the label,and the noOfPossibleLabel are 3.
       // DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,4,3);
///////////////added////////
        int numLinesToSkip = 0;
        String delimiter = ",";
        RecordReader rr = new CSVRecordReader(numLinesToSkip,delimiter);
        rr.initialize(new FileSplit(new ClassPathResource("iris_data_training.txt").getFile()));

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 4;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        int numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        //int batchSize = 50;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
        //creating for one batch-84 data in the training set.
        org.nd4j.linalg.dataset.api.iterator.DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,84,labelIndex,numClasses);


        RecordReader rrTest = new CSVRecordReader(numLinesToSkip,delimiter);
        rrTest.initialize(new FileSplit(new ClassPathResource("iris_data_testing.txt").getFile()));

        //Load the test/evaluation data:
        //RecordReader rrTest = new CSVRecordReader();
        //rrTest.initialize(new FileSplit(new File("C:/Users/Lakini/Desktop/CSVData/iris_data_testing.csv")));
        //creating for one batch-65 data in the training set.
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,65,labelIndex,numClasses);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list(2)
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax").weightInit(WeightInit.XAVIER)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1)));

        for ( int n = 0; n < nEpochs; n++) {
            model.fit( trainIter );
        }

        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        while(testIter.hasNext()){
            DataSet t = testIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features,false);

            eval.eval(lables, predicted);

        }

        //Print the evaluation statistics
        System.out.println(eval.stats());

        System.out.println("End of Evaluation session!!!!");

        //------------------------------------------------------------------------------------
        //Training is complete. Code that follows is for plotting the data & predictions only

        //Plot the data://based on 2 axis.//but have to think about more columns
//        double xMin = 0;
//        double xMax = 8.0;
//        double yMin = 0;
//        double yMax = 2;
//
//        //changed up to here
//        //Let's evaluate the predictions at every point in the x/y input space
//        int nPointsPerAxis = 1;
//        double[][] evalPoints = new double[nPointsPerAxis*nPointsPerAxis][1];
//        int count = 0;
//        for( int i=0; i<nPointsPerAxis; i++ ){
//            for( int j=0; j<nPointsPerAxis; j++ ){
//                double x = i * (xMax-xMin)/(nPointsPerAxis-1) + xMin;
//                double y = j * (yMax-yMin)/(nPointsPerAxis-1) + yMin;
//
//                evalPoints[count][0] = x;
//                evalPoints[count][1] = y;
//
//                count++;
//            }
//        }
//
//        INDArray allXYPoints = Nd4j.create(evalPoints);
//        INDArray predictionsAtXYPoints = model.output(allXYPoints);
//
//        System.out.println("Heloooooo");
//
//        //changed up to here
//        //Get all of the training data in a single array, and plot it:
//        rr.initialize(new FileSplit(new File("linear_data_train.csv")));
//        rr.reset();
//        int nTrainPoints = 84;
//        trainIter = new RecordReaderDataSetIterator(rr,nTrainPoints,4,3);
//        DataSet ds = trainIter.next();
//        PlotUtil.plotTrainingData(ds.getFeatures(), ds.getLabels(), allXYPoints, predictionsAtXYPoints, nPointsPerAxis);
//
//
//        //Get test data, run the test data through the network to generate predictions, and plot those predictions:
//        rrTest.initialize(new FileSplit(new File("iris_data_testing.txt")));
//        rrTest.reset();
//        int nTestPoints = 65;
//        testIter = new RecordReaderDataSetIterator(rrTest,nTestPoints,4,3);
//        ds = testIter.next();
//        INDArray testPredicted = model.output(ds.getFeatures());
//        PlotUtil.plotTestData(ds.getFeatures(), ds.getLabels(), testPredicted, allXYPoints, predictionsAtXYPoints, nPointsPerAxis);
//
//        System.out.println("****************Example finished********************");
    }
}
