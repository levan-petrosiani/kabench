import requests
import json
import time
from datetime import datetime

OLLAMA_URL_CHAT = "http://localhost:11434/api/chat"
OLLAMA_URL_GEN  = "http://localhost:11434/api/generate"
MODELS = ["qwen2.5:14b"]

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
filename = f"{timestamp}-test.md"

test_cases = [
    "მე გუშინ წავედი მაღაზიაში და ვიყიდე პური მერე მოვედი სახლში.",
    "მან მითხრა რომ მოვა მაგრამ არ მოვიდა.",
    "ბავშვიები დარბიან ეზოში.",
    "ჩვენ წავედით ქალაქში სადაც ბევრი ხალხი იყო.",
    "პროექტი რომელიც ჩვენ დავიწყეთ წარმატებული იქნება.",
    "კაცი დაინახა მეგობარი.",
    "ხუთი ბავშვები მოვიდნენ.",
    "მან მითხრა რომ მოვა.",
    "მან გაანალიზა რომ მთელი მისი ცხოვრება ფუჭი ყოფილა",
    "კაცი დაინახა მეგობარი და გაუღიმა.",
    "ათი ბავშვები მოვიდნენ ეზოში.",
    "მე მინდა რომ წავიდე.",
    "ავტომობილი როგორიც გუშინ ვიყიდეთ ძალიან სწრაფია.",
    "ბალახი გათიბა კაცმა.",
    "არც ის მოვიდა არც მისი მეგობარი.",
    "მან მითხრა მოვალო.",
    "მაგიდაზეზე დევს წიგნი.",
    "ეს არის ყველაზე უფრო საუკეთესო ვარიანტი.",
    "დედამ აჩუქა შვილი სათამაშო.",
]

def load_model(model):
    """Triggers Ollama to load the model into memory without running a prompt."""
    print(f"--- Loading model {model} into memory... ---")
    try:
        # Sending an empty prompt with keep_alive ensures the model stays in VRAM
        requests.post(OLLAMA_URL_GEN, json={"model": model, "keep_alive": "5m"}, timeout=60)
        print("--- Model loaded and ready. ---")
    except Exception as e:
        print(f"Error loading model: {e}")

def run_test(model, text):
    prompt = f"""
 შენ ხარ ქართული ენის ექსპერტი კორექტორი. შენი ერთადერთი ამოცანაა გაასწორო მხოლოდ აშკარა ორთოგრაფიული, გრამატიკური და პუნქტუაციის შეცდომები მინიმალური ცვლილებებით.

წესები (აუცილებლად დაიცავი ყველა):
- გაასწორე მხოლოდ შეცდომები: ორთოგრაფია (მაგ. სჰ → შ), გრამატიკა (შეთანხმება, შემთხვევები, დროები), პუნქტუაცია (მძიმეები რომ-თან, თუმცა-თან, ჩასმული სიტყვებით).
- არ შეცვალო სწორი სიტყვები, დროები ან მნიშვნელობა.
- არ დაამატო ან ამოიღო სიტყვები, თუ ეს აბსოლუტურად აუცილებელი არ არის გრამატიკისთვის (მაგ. ორმაგი 'და' ამოიღე).
- თუ ტექსტი უკვე სწორია, დააბრუნე უცვლელად.
- უპასუხე მხოლოდ და მხოლოდ გასწორებული ტექსტით. არც ახსნა, არც "Output:", არც "Explanation:", არც შენიშვნა, არც ბრჭყალები, არც არაფერი სხვა.

ფიქრის პროცესი (გააკეთე გონებაში ნაბიჯ-ნაბიჯ, მაგრამ არ დაწერო):

1. წაიკითხე ტექსტი და გამოყავი შესაძლო შეცდომები: ორთოგრაფია, შეთანხმება (სუბიექტი-ზმნა), შემთხვევები (ერგატიული წარსულში), დროები (აგლუტინაცია), პუნქტუაცია (მძიმეები რომ-თან, ჩასმულებთან, კონიუნქციებთან), რედუნდანტობა (ორმაგი სიტყვები), რიცხვები (რიცხვი + მხოლობითი).
2. თუ გაურკვეველია, მენტალურად თარგმნე ინგლისურში (როგორც მაღალრესურსულ ენაში), გაასწორე იქ გრამატიკა, შემდეგ უკან თარგმნე ქართულში მინიმალური ცვლილებებით (არ შეცვალო მნიშვნელობა).
3. გააკეთე მინიმალური ცვლილებები და შეამოწმე, რომ მნიშვნელობა უცვლელი დარჩეს.
4. ბოლოს, გამოიტანე მხოლოდ გასწორებული ტექსტი.

მაგალითები:

მაგალითი 1 (შეთანხმება):
Input: ბავშვებმა დაინახა კატა და გაიქცნენ.
Output: ბავშვებმა დაინახეს კატა და გაიქცნენ.

მაგალითი 2 (პუნქტუაცია რომ-თან):
Input: მან თქვა რომ მოვა მაგრამ არ მოვიდა.
Output: მან თქვა, რომ მოვა, მაგრამ არ მოვიდა.

მაგალითი 3 (ზმნის დრო/აგლუტინაცია):
Input: ისინი სახლს აშენებენ წინა წელს.
Output: ისინი სახლს აშენებდნენ წინა წელს.

მაგალითი 4 (შემთხვევა - ერგატიული):
Input: კაცი დაინახა მეგობარი.
Output: კაცმა დაინახა მეგობარი.

მაგალითი 5 (რედუნდანტობა):
Input: ჩვენ წავედით ქალაქში სადაც ბევრი ხალხი იყო და და დავისვენეთ.
Output: ჩვენ წავედით ქალაქში, სადაც ბევრი ხალხი იყო და დავისვენეთ.

მაგალითი 6 (ჩასმული სიტყვები/პუნქტუაცია):
Input: ეს რა თქმა უნდა სიმართლეა.
Output: ეს, რა თქმა უნდა, სიმართლეა.

მაგალითი 7 (რიცხვები + მხოლობითი):
Input: ხუთი მეგობრები მოვიდნენ.
Output: ხუთი მეგობარი მოვიდა.

მაგალითი 8 (კონიუნქციები):
Input: არა მხოლოდ ის არამედ მისი ძმაც მოვიდა.
Output: არა მხოლოდ ის, არამედ მისი ძმაც მოვიდა.

მაგალითი 9 (ორთოგრაფია/ფონეტიკური):
Input: სჰეცდომა არის სჰესასწორებელი.
Output: შეცდომა არის შესასწორებელი.

მაგალითი 10 (სწორი ტექსტი - უცვლელი):
Input: კომპიუტერული ტექნოლოგიები და ინოვაციები მნიშვნელოვანია.
Output: კომპიუტერული ტექნოლოგიები და ინოვაციები მნიშვნელოვანია.

ახლა გაასწორე:
Input: {text}

Output: 
"""
    
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": text}],
        "stream": False,
        "options": {"temperature": 0, "stop": ["Explanation:", "Output:", "Note:"]}
    }
    
    start_time = time.time()
    try:
        r = requests.post(OLLAMA_URL_CHAT, json=payload, timeout=30)
        end_time = time.time()
        
        duration = end_time - start_time
        response = r.json().get('message', {}).get('content', '').strip()
        clean = response.split("Output:")[-1].split("Explanation:")[0].strip()
        
        return clean, duration
    except Exception as e:
        return f"Error: {str(e)}", 0

def escape_pipe(text):
    return text.replace("|", "\\|")

# Main Execution
results = []
for model in MODELS:
    load_model(model) # Warm up phase
    
    print(f"Starting benchmark for {model}...")
    for case in test_cases:
        print(f"Processing: {case[:30]}...")
        corrected, duration = run_test(model, case)
        results.append({
            "original": case,
            "model": model,
            "corrected": corrected,
            "duration": duration
        })

# Generate Markdown with Speed Metrics
with open(filename, "w", encoding="utf-8") as f:
    f.write(f"# Georgian LLM Benchmark - Speed & Accuracy ({timestamp})\n\n")
    f.write("| Original Text | Corrected Version | Speed (sec) |\n")
    f.write("| :--- | :--- | :--- |\n")

    total_time = 0
    for res in results:
        total_time += res['duration']
        f.write(f"| {escape_pipe(res['original'])} | {escape_pipe(res['corrected'])} | {res['duration']:.2f}s |\n")

    avg_time = total_time / len(test_cases)
    f.write(f"\n\n**Total Benchmark Time:** {total_time:.2f}s  \n")
    f.write(f"**Average Speed per Case:** {avg_time:.2f}s")

print(f"\nBenchmark complete! Results saved to {filename}")