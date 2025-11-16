import requests
import json
from src.st53.inference_st53 import ST53Predictor
from typing import Dict, List, Any


def run_test_predictions() -> Dict[str, Any]:
    """Run comprehensive tests and collect results for LLM analysis"""
    predictor = ST53Predictor("models")
    
    test_results = {
        "model_info": {
            "window_size": predictor.window_size,
            "scaler_min": float(predictor.scaler.data_min_[0]),
            "scaler_max": float(predictor.scaler.data_max_[0]),
        },
        "tests": []
    }
    
    # Real Cenovus SAGD production data from API examples
    # MODEL TRAINED ON LARGE OPERATIONS ONLY (>15,000 m³/month average)
    # Includes: Christina Lake, Foster Creek, Jackfish, Surmont, Cold Lake, Firebag, MEG Christina Lake
    test_cases = [
        {
            "name": "Cenovus Christina Lake",
            "input": [38440.48, 38453.59, 38345.22, 23973.98, 40922.27, 40339.68, 38701.56, 37223.34],
            "expected_range": [25000, 45000],
            "description": "Largest Cenovus SAGD operation (~37,000-41,000 m³/month)"
        },
        {
            "name": "Cenovus Foster Creek",
            "input": [30717.12, 32122.67, 30897.38, 30137.87, 29907.24, 31504.45, 30909.04, 30768.31],
            "expected_range": [25000, 35000],
            "description": "Large consistent operation (~30,000-32,000 m³/month)"
        },
        {
            "name": "Suncor Firebag",
            "input": [35622.35, 36088.92, 35891.63, 36188.79, 36895.32, 37678.31, 38245.55, 37188.41],
            "expected_range": [30000, 42000],
            "description": "Major Suncor operation (~35,000-38,000 m³/month)"
        },
    ]
    
    for test in test_cases:
        prediction = predictor.predict(test["input"])
        input_avg = sum(test["input"]) / len(test["input"])
        input_trend = test["input"][-1] - test["input"][0]
        
        test_results["tests"].append({
            "name": test["name"],
            "description": test["description"],
            "input": test["input"],
            "input_average": round(input_avg, 2),
            "input_trend": round(input_trend, 2),
            "last_value": test["input"][-1],
            "prediction": round(prediction, 2),
            "expected_range": test["expected_range"],
            "in_expected_range": test["expected_range"][0] <= prediction <= test["expected_range"][1],
            "deviation_from_avg": round(prediction - input_avg, 2),
        })
    
    return test_results


def analyze_with_ollama(test_results: Dict[str, Any], model: str = "gemma3:1b") -> str:

    
    prompt = f"""You are analyzing bitumen production forecasts for Cenovus Energy's SAGD operations in Alberta.

MODEL: LSTM neural network trained on ST53 Alberta production data (range: 0-40,922 m³/month)

PRODUCTION DATA ANALYZED:
{json.dumps(test_results['tests'], indent=2)}

YOUR TASK:
Write a clear, concise report for operations managers. Focus on:

1. **PREDICTIONS SUMMARY** 
   Use bullet points (with *) for each site. Format EXACTLY like this:
   * Site Name: Current production trend: XXXX m³. Model's prediction for next month: YYYY m³. Verdict: Production expected to go UP/DOWN/REMAIN STABLE.
   
   Example format:
   * Christina Lake: Current production trend: -1217.14. Model's prediction for next month: 33719.36. Verdict: Production expected to go Down.

2. **RECOMMENDATION**
   One sentence: What should operations teams focus on?

IMPORTANT: 
- Use bullet points (*), NOT tables
- Use simple, direct language
- Compare prediction to the 8-month average to determine if it's going up/down/stable
- Be specific about which sites are performing well and which need attention
- Keep it concise - operations managers are busy!

Write the report now:
"""
    
    try:
        print("🤖 Sending test results to Ollama LLM for analysis...")
        print(f"   Using model: {model}")
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more focused analysis
                }
            },
            timeout=120  # 2 minute timeout
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: Ollama returned status code {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return """
❌ ERROR: Could not connect to Ollama.

Please ensure Ollama is running:
1. Install Ollama: https://ollama.ai/
2. Pull a model: ollama pull llama3.2
3. Start server: ollama serve

Or use LM Studio: https://lmstudio.ai/
"""
    except Exception as e:
        return f"❌ ERROR: {str(e)}"


def main():
    """Main function to run tests and analyze with LLM"""
    print("="*70)
    print("ST53 MODEL ANALYSIS WITH LOCAL LLM")
    print("="*70)
    
    # Run tests
    print("\n📊 Running model tests...")
    test_results = run_test_predictions()
    
    print(f"\n✅ Completed {len(test_results['tests'])} test cases")
    print("\n📋 Test Summary:")
    for test in test_results["tests"]:
        status = "✅" if test["in_expected_range"] else "❌"
        print(f"{status} {test['name']}: {test['prediction']:.2f} m³ "
              f"(expected: {test['expected_range'][0]}-{test['expected_range'][1]})")
    
    # Analyze with LLM
    print("\n" + "="*70)
    analysis = analyze_with_ollama(test_results)
    
    print("\n🤖 LLM ANALYSIS:\n")
    print(analysis)
    print("\n" + "="*70)
    
    # Save results to file
    output_file = "model_analysis_report.json"
    with open(output_file, 'w') as f:
        json.dump({
            "test_results": test_results,
            "llm_analysis": analysis
        }, f, indent=2)
    
    print(f"\n💾 Full report saved to: {output_file}")


if __name__ == "__main__":
    main()
