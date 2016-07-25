package org.deeplearning4j.examples.feedforward.classification;

import jdk.nashorn.internal.parser.JSONParser;
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
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;
import java.util.Collections;

public class MLPClassifierLinearIrisData_test {

    public String createFeedForwardNetwork() throws IOException, InterruptedException {
        int seed = 123;
        double learningRate = 0.01;
        int nEpochs = 1;
        int numInputs = 4;
        int numOutputs = 3;
        int numHiddenNodes = 20;
        int numLinesToSkip = 0;
        String delimiter = ",";

        RecordReader rr = new CSVRecordReader(numLinesToSkip,delimiter);
        rr.initialize(new FileSplit(new ClassPathResource("iris_data_training.txt").getFile()));

        int labelIndex = 4;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        int numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        org.nd4j.linalg.dataset.api.iterator.DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,84,labelIndex,numClasses);
        RecordReader rrTest = new CSVRecordReader(numLinesToSkip,delimiter);
        rrTest.initialize(new FileSplit(new ClassPathResource("iris_data_testing.txt").getFile()));
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

        //eval.stats()

        System.out.println("End of Evaluation session!!!!");
        return eval.stats();

    }

}
