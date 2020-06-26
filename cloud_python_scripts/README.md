### Sentiment-Analysis-PoC Overview

The objective for this project is to create a sentiment analysis application that provides valuable insights out of satisfaction surveys, current version takes in count a public dataset provided by government organization/entities, in which we can find comments done to different kinds of Mexican public entities online platforms, such as IMSS, SAT, SEP, among others.

So what can we extract out of that data?, in there we can get if customers are fine with the service provided or not, comments about how can we improve X online system, the date in which these comments were done and to which Mexican organization/entity.

The initial version of the application lives on **Google Cloud**, specifically... on a **VM Instance**.
First, a Python script takes the mentioned dataset stored on **Google Cloud Storage** and by using ¿Google Cloud's Natural Language API and **Pandas Framework**, we're processing and analyzing the comments done by people to X organization's online platform, then we are rating if the comments done are positive, negative or neutral and storing the results back to Google Cloud Storage.

Ideally we want this to grow and be modular enough to work for different kind of surveys, but this is the first step, and looks like a good way to start.

### Repository Directory Structure
```
+--
+-- _Sentiment-Analysis-PoC
  +-- _python_scripts
  |   +-- _analyze_organization_sentiment.py
  +-- README.md
```

### Project Dependencies
Currently the project's major dependency is for no one's surprise... Google Cloud Platform.
But if we want to get more specific, here's the list of Cloud Services and API's being used and which we could consider as dependencies for the project.
- Google Cloud Services
  - VM Instance
  - Google Cloud Storage
- Code Dependencies
  - Python (2.7)
  - Natural Language API
  - Google Cloud Storage API
  - Pandas Framework for Python

### Running the application
In order to run this script, we have to make sure we have a VM instance with the necessary dependencies installed, for future usage, it would be a good idea to have a script that automates this setup, so people avoids reading this manual as much as possible.

Once you have the setup on the VM Instance,
you need to pass the python script inside 'python_scripts' to your terminal and run it with some arguments, as follows:
```
Command:
python analyze_organization_sentiment.py [CSV_FILE_ON_GCS]
[ORGANIZATION/ENTITY] [OUTPUT_BUCKET] [OUTPUT_CSV]

E.g.
python analyze_organization_sentiment.py
gs://dankyboy-bucket/sentiment-analysis-PoC/sample.csv SRE
dankyboy-bucket results.csv
```


