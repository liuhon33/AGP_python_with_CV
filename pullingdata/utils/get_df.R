# utils/get_df.R  -----------------------------------------------------------
get_df <- function(field_list,
                   instance_list = NULL,
                   params_path   = "./utils/params.yaml") {
 
  # 1. Spark bootstrap ------------------------------------------------------
  Sys.setenv(SPARK_HOME = "/external/rprshnas01/netdata_kcni/dflab/tools/general/distributed-computing/spark/3.5.0/")
  library(yaml)
  library(magrittr)
  library(SparkR,
          lib.loc = file.path(Sys.getenv("SPARK_HOME"), "R", "lib"))
 
  sparkR.session(
    master = "local[12]",
    sparkConfig = list(
      spark.driver.memory   = "10g",
      spark.executor.memory = "10g",
      spark.sql.extensions  = "io.delta.sql.DeltaSparkSessionExtension",
      spark.sql.catalog.spark_catalog =
        "org.apache.spark.sql.delta.catalog.DeltaCatalog"
    ),
    sparkPackages = "io.delta:delta-spark_2.12:3.0.0"
  )
 
  # 2. Load configuration ---------------------------------------------------
  params <- read_yaml(params_path)
  if (is.null(instance_list))
    instance_list <- params$ukbiobank$main$instances
 
  # 3. Read Delta table lazily ---------------------------------------------
  df_ukb_delta <- read.df(params$ukbiobank$main$delta, source = "delta")
 
  df_subset <- df_ukb_delta                           %>%
    select(c(list("eid", "instance"), field_list))    %>%
    filter(df_ukb_delta$instance %in% instance_list)
 
  # 4. Attach human-readable column names -----------------------------------
  dict <- read.df(params$ukbiobank$main$dictionary,
                  source = "csv", delimiter = "\t",
                  header = TRUE, inferSchema = TRUE)
 
  mapping <- dict                                   %>%
    select(c("FieldID", "Field"))                   %>%
    filter(dict$FieldID %in% colnames(df_subset))   %>%
    withColumn("pretty",
               expr("concat(`Field`, ' (FieldID: ', `FieldID`, ')')"))
 
  names(df_subset)[match(collect(mapping)$FieldID,
                         names(df_subset))] <- collect(mapping)$pretty
 
  # 5. Materialise result in R driver & clean up ----------------------------
  out <- collect(df_subset)
  sparkR.stop()
  return(out)
}