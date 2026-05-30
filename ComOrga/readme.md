# **ComOrga System Operation Manual**

## **1\. Software Overview**

ComOrga is a high-content microscopic image data analysis platform integrating tools such as CellProfiler, RStudio, and Python. It supports 3D image stack reconstruction, texture feature and interaction parameter extraction, and downstream analyses including comprehensive t-SNE / PCA scoring.

The system integrates a complete workflow from low-level image preprocessing to high-level statistical analysis, including:

* 2D slice spatial reconstruction  
* CellProfiler-based single-cell texture and interaction feature extraction  
* Metadata-based data cleaning and batch integration  
* Visualization and statistical testing based on t-SNE (non-linear manifold learning) and PCA (Principal Component Analysis)

## **2\. Interface Architecture**

The system adopts a three-column logical layout to support the data exploration and analysis process:

1. **Left Sidebar \- Control and Log Tree**  
   Configure environment variables, mount the global working directory, select Pipeline nodes, and view real-time operation logs.  
2. **Middle Panel \- Algorithm and Mapping Configuration**  
   Provides control options such as feature selection and data orientation based on the current analysis node.  
3. **Right Panel \- Multi-modal Analysis Dashboard**  
   * **Plot & Stats**: Displays t-SNE / PCA results, supporting interactive testing.  
   * **CSV Data Preview**: Previews the data headers to verify data orientation and quality (QC).

## **3\. Pre-execution Environment Deployment**

### **3.1 Environment Variable Verification**

When running the application for the first time or changing devices, ensure the system is correctly linked to the underlying computing engines.

1. Click **Environment Path Settings** in the left control tree.  
2. The system will automatically search for key engine paths. Please verify if the following paths are correct in the pop-up dialog:  
   * **CellProfiler.exe Path**: Used for image feature extraction (recommended version 4.2.6).  
   * **CP Pipeline (.cppipe)**: Pipeline for cell segmentation and feature extraction.  
   * **Rscript.exe Path** and **tsne\_merged.R Path**: The underlying manifold learning engine for t-SNE (recommended R version 4.5.0).

Note: You can use the file icon button to manually select a specific file, or use the magnifying glass button to auto-scan a folder.

### **3.2 Set Global Metadata Grouping Basis**

Find the **Global Group By:** option in the left panel. This parameter determines how downstream data is grouped based on well plate naming:

* **Letter**: Splits by the first letter of the well (e.g., extracting 'B' from 'B02'). This is suitable for row-to-row control experiments.  
* **Number**: Splits by the numbers of the well (e.g., extracting '02' from 'B02'). This is suitable for column-to-column control experiments.

## **4\. Core Analysis Pipeline Operation Guide**

### **4.0 Global Directory Mounting (Initialization)**

1. Click **Select Raw 2D Folder / Auto-Link Paths** at the top left.  
2. Select the root directory containing the raw 2D TIFF microscopic images.  
3. The system will use this directory as an anchor to automatically derive and populate the input/output directories for the subsequent 5 steps, forming a closed-loop workflow.

### **Step 1: 1\. concattif (2D to 3D)**

* Based on the image naming conventions (row, column, field of view, fluorescence channel), discrete 2D slices are reconstructed into a 3D spatial tensor along the Z-axis order.  
* Verify the **3D TIF Output Dir** directory. Once confirmed, click **Run Current Task** to execute spatial reconstruction.

### **Step 2: 2\. CellProfiler Analysis**

* Calls the independent CellProfiler engine, inputs the 3D image sequence, and loads the specified analysis pipeline. This achieves unbiased segmentation of organelles (e.g., endoplasmic reticulum, mitochondria) and quantification of high-dimensional colocalization features (Manders' coefficient, Pearson's correlation coefficient, etc.).  
* **Note**: This step consumes substantial computing resources. After confirming the paths are correct, click to execute.  
* The pipeline design can be adjusted according to experimental objectives.

### **Step 3: 3\. Extract & Group Parameters**

* The raw CellProfiler output data volume is large. This node provides three filtering methods in the **Extract Mode** dropdown: **Preset Parameters (13 Core Metrics)**, **All Parameters (Keep Everything)**, and **Custom Selection**.  
* The system automatically uses the first two "\_" separators in the parameter names as classification standards. You can search for parameter names to filter and set the action for selected parameters to **Include Selected** or **Exclude Selected**.  
* For image parameters, the system automatically calculates the mean. For single-cell parameters, the system automatically removes NA values.

### **Step 4 & Step 5: 4\. Aggregated t-SNE Reduction and 5\. Aggregated PCA Scoring**

These two steps share the same data aggregation logic. The following operation instructions apply to both Step 4 (t-SNE) and Step 5 (PCA).

**Key Operation Steps:**

1. **Define Data Tensor Orientation**  
   Observe the **CSV Data Preview** tab on the right.  
   * If each row represents a single cell/sample and each column represents a feature, select **Rows are Samples (Standard)** under Data Orientation.  
   * Conversely, if transposition is needed, select **Columns are Samples (Transpose)**.  
2. **Filter Analysis Features**  
   In the **Select Params (Tick=Use, Click=Preview):** list in the middle panel, check the features to be included in the dimensionality reduction calculation. Unchecked features will be excluded to reduce noise interference.  
3. **Adjust Algorithm Parameters**  
   **Step 4 (t-SNE) requires configuring:**  
   * **Perplexity**: Controls the local neighborhood size that t-SNE focuses on.  
     * Default is 30 for single-cell samples.  
     * Default is 5 for single-well samples.  
   * **Point Size**:  
     * Default is 1.2 for single-cell samples, 3 for single-well samples.  
   * **Point Transparency**:  
     * Default is 0.6 for single-cell samples, 1 for single-well samples.

**Step 5 (PCA)** requires no additional parameter configuration beyond standard Point Size and Point Transparency inputs.

4. **Construct Mapping and Aggregation Configuration Matrix (Core Operation)**  
   The **Mapping & Aggregation Configuration** panel recombines the discrete data split from Step 3 into a total feature matrix for the dimensionality reduction algorithms.  
   * **Prefix**: Enter the physical prefix of the file. (The "Detected Prefixes in Folder" label at the top of the panel lists available prefixes, e.g., 02, 03).  
   * **Name**: Specify a label for this cohort in the visualization results (e.g., Control, Model).  
   * **Count**: Set the number of samples from this cohort to participate in the calculation. To maintain sampling balance, it is recommended to keep the quantity consistent across groups. Entering 0 means using all samples in the file with that prefix.  
   * **Color**: Click the color block to specify a color scheme for each biological group.

Note: Unchecking the checkbox at the beginning of a row removes the cohort from the current analysis (eliminating the need for frequent additions or deletions). Cohorts with empty prefixes are automatically ignored.

5. **Execute Analysis**  
   * Click **Run Current Task** to execute.  
   * After **Step 5 (Aggregated PCA Scoring)** is completed, the system not only generates a boxplot for comprehensive principal component scoring but also automatically performs Analysis of Variance (ANOVA) and pairwise t-tests. Significance markers (e.g., \*\*\*, ns) are rendered directly into the output vector graphics (SVG / PNG).