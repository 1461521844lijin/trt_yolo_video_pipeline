import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class MorphologyDemo2 {
    private static final String[] MORPH_OP = { "Opening", "Closing", "Gradient", "Top Hat", "Black Hat" };
    private static final int[] MORPH_OP_TYPE = { Imgproc.MORPH_OPEN, Imgproc.MORPH_CLOSE,
            Imgproc.MORPH_GRADIENT, Imgproc.MORPH_TOPHAT, Imgproc.MORPH_BLACKHAT };
    private static final String[] ELEMENT_TYPE = { "Rectangle", "Cross", "Ellipse" };
    private static final int MAX_KERNEL_SIZE = 21;
    private Mat matImgSrc;
    private Mat matImgDst = new Mat();
    private int morphOpType = Imgproc.MORPH_OPEN;
    private int elementType = Imgproc.CV_SHAPE_RECT;
    private int kernelSize = 0;
    private JFrame frame;
    private JLabel imgLabel;

    public MorphologyDemo2(String[] args) {
        String imagePath = args.length > 0 ? args[0] : "../data/LinuxLogo.jpg";
        matImgSrc = Imgcodecs.imread(imagePath);
        if (matImgSrc.empty()) {
            System.out.println("Empty image: " + imagePath);
            System.exit(0);
        }

        // Create and set up the window.
        frame = new JFrame("Morphology Transformations demo");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        // Set up the content pane.
        Image img = HighGui.toBufferedImage(matImgSrc);
        addComponentsToPane(frame.getContentPane(), img);
        // Use the content pane's default BorderLayout. No need for
        // setLayout(new BorderLayout());
        // Display the window.
        frame.pack();
        frame.setVisible(true);
    }

    private void addComponentsToPane(Container pane, Image img) {
        if (!(pane.getLayout() instanceof BorderLayout)) {
            pane.add(new JLabel("Container doesn't use BorderLayout!"));
            return;
        }

        JPanel sliderPanel = new JPanel();
        sliderPanel.setLayout(new BoxLayout(sliderPanel, BoxLayout.PAGE_AXIS));

        JComboBox<String> morphOpBox = new JComboBox<>(MORPH_OP);
        morphOpBox.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                @SuppressWarnings("unchecked")
                JComboBox<String> cb = (JComboBox<String>)e.getSource();
                morphOpType = MORPH_OP_TYPE[cb.getSelectedIndex()];
                update();
            }
        });
        sliderPanel.add(morphOpBox);

        JComboBox<String> elementTypeBox = new JComboBox<>(ELEMENT_TYPE);
        elementTypeBox.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                @SuppressWarnings("unchecked")
                JComboBox<String> cb = (JComboBox<String>)e.getSource();
                if (cb.getSelectedIndex() == 0) {
                    elementType = Imgproc.CV_SHAPE_RECT;
                } else if (cb.getSelectedIndex() == 1) {
                    elementType = Imgproc.CV_SHAPE_CROSS;
                } else if (cb.getSelectedIndex() == 2) {
                    elementType = Imgproc.CV_SHAPE_ELLIPSE;
                }
                update();
            }
        });
        sliderPanel.add(elementTypeBox);

        sliderPanel.add(new JLabel("Kernel size: 2n + 1"));
        JSlider slider = new JSlider(0, MAX_KERNEL_SIZE, 0);
        slider.setMajorTickSpacing(5);
        slider.setMinorTickSpacing(5);
        slider.setPaintTicks(true);
        slider.setPaintLabels(true);
        slider.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                JSlider source = (JSlider) e.getSource();
                kernelSize = source.getValue();
                update();
            }
        });
        sliderPanel.add(slider);

        pane.add(sliderPanel, BorderLayout.PAGE_START);
        imgLabel = new JLabel(new ImageIcon(img));
        pane.add(imgLabel, BorderLayout.CENTER);
    }

    private void update() {
        Mat element = Imgproc.getStructuringElement(elementType, new Size(2 * kernelSize + 1, 2 * kernelSize + 1),
                new Point(kernelSize, kernelSize));

        Imgproc.morphologyEx(matImgSrc, matImgDst, morphOpType, element);
        Image img = HighGui.toBufferedImage(matImgDst);
        imgLabel.setIcon(new ImageIcon(img));
        frame.repaint();
    }

    public static void main(String[] args) {
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Schedule a job for the event dispatch thread:
        // creating and showing this application's GUI.
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new MorphologyDemo2(args);
            }
        });
    }
}
