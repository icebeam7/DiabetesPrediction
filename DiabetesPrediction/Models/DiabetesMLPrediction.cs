using Microsoft.ML.Data;

namespace DiabetesPrediction.Models
{
    public class DiabetesMLPrediction : Patient
    {
        [ColumnName("PredictedLabel")]
        public float Prediction { get; set; }

        public float Probability { get; set; }

        public float[] Score { get; set; }
    }
}
