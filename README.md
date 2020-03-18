# CHBLab
<p><strong><u>Computational Human Behavior Lab</u></strong></p>
<h2><p><strong><u>Project Proposal</u></strong></h2>
<p><strong><u>Students:</u></strong></p>
<p style="text-indent: 40px">Elie Abboud - 315475889</p>
<p style="text-indent: 40px">Daniel Shalam - 205745052</p>
<br>
<p><strong><u>Title:</u></strong> Analyzing MATH Problem Solving</p>
<p><strong><u>End Goal:</u></strong></p>
<p>A system that will collect data from eye-tracker of students who are solving math problems and give a binary classification for the students: <strong>talented</strong> or <strong>high-achieving</strong>.</p>
</br>
<ul>
<li>Training:
<ol>
<li>Given raw eye-tracker data, clean and transform the data (coordinates, fixations, saccades, etc.).</li>
<li>Learn and classify from given data using a probabilistic model.</li>
</ol>
</li>
<li>Run-Time: Input of <strong>new</strong> unprecedented eye-tracker data and classify the new data as <strong>talented</strong> or <strong>high-achieving</strong>.</li>
</ul>
<p><strong><u>Methods:</u></strong></p>
<p>The system might rely on:</p>
<ul>
<li>Project will be implemented in Python.</li>
<li>PyGaze which will help collect and convert raw eye-tracker data to helpful classified data<br /> (fixations and saccades).</li>
<li>PyMVPA: which will use Machine Learning (specifically SVM) to help with data classifications.</li>
<li>TensorFlow: which will be used as a basis for ML and DL algorithms.</li>
<li>Large Data Sets: That were provided by previous companies/researchers that were made public for additional learning.</li>
<li>Multi-match gaze &ndash; which will find patterns of data between different eye paths<br /> (using multi-match algorithm).</li>
</ul>
<p><strong><u>Expected Challenges:</u></strong></p>
<ul>
<li>The system will need to work on various illumination settings.</li>
<li>Accurate data collection (setup).</li>
<li>Limited Learning (need more samples) &ndash; overfitting and outliers.</li>
<li>Feature Selection.</li>
<li>Calibration</li>
</ul>
<p><strong><u>Possible Solutions to Challenges:</u></strong></p>
<ul>
<li>Normalize input images, (color and lighting) and perform image enhancement or Histogram Equalization.</li>
<li>Add IR Lights to camera setup (human eyes shouldn&rsquo;t perceive this light)</li>
<li>Use additional data sets from other problems to help further classify subjects, <strong>if possible</strong>.</li>
<li>Look at common locations participants look and identify key-areas of interest. Consider if subject is looking aside from the screen.</li>
<li>Use Quantile regression &ndash; it&rsquo;s a model that designed to be robust to outliers. (<a href="https://en.wikipedia.org/wiki/Quantile_regression">https://en.wikipedia.org/wiki/Quantile_regression</a>)</li>
<li>Regularization Techniques (found in Tensorflow) &ndash; helps reduce the effect of overfitting the data.</li>
<li>Auto-encoders &ndash; for feature reduction (avoid curse of dimensionality).</li>
</ul>
<p><strong><u>Other Methods that were Considered:</u></strong></p>
<ul>
<li>Blackshift Analytics as an analysis tool of perceived data (video, and fixations).</li>
<li>OpenCV methods.</li>
<li>Ogama</li>
<li><a href="https://medium.com/datadriveninvestor/small-data-deep-learning-ai-a-data-reduction-framework-9772c7273992">https://medium.com/datadriveninvestor/small-data-deep-learning-ai-a-data-reduction-framework-9772c7273992</a></li>
</ul>
<p><strong><u>TimeLine Plan:</u></strong></p>
<ul>
<li>Library Installation + Get Familiar with libraries &ndash; 1 week.</li>
<li>Decide on System Architecture Plan + ML/DL Algorithms used + Initial System Implementation &ndash; 1 week</li>
<li>Continued System Implementation + Testing on sample inputs &ndash; 2 weeks</li>
<li>Mid proposal demo</li>
<li>Testing on real collected data + Continued System Implementation (Optimization) - 7 weeks</li>
<li>Rap up - Rap up the system for export:&nbsp;&nbsp;- 1 week
<ul>
<li>Clean code, add comments and documentation</li>
<li>Write up installation instructions</li>
<li>Write up user instructions</li>
<li>Write up Project Report</li>
</ul>
</li>
</ul>
<p>&nbsp;</p>
