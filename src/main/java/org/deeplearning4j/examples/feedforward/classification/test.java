package org.deeplearning4j.examples.feedforward.classification;

/**
 * Created by Lakini on 7/25/2016.
 */
public class test {

    public static void main(String[] args) throws Exception {

        MLPClassifierLinearIrisData_test x=new MLPClassifierLinearIrisData_test();
        String y= x.createFeedForwardNetwork();
        System.out.println("From Tset"+y);
    }
}
