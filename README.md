# Command Line:

    # Local
    export MLFLOW_TRACKING_URI='http://localhost:5000'
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    
    # Managed (Azure)
    export MLFLOW_TRACKING_URI=databricks
    export DATABRICKS_HOST="..."
    export DATABRICKS_TOKEN="..."
