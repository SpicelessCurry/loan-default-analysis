import re

with open('app.py', 'r', encoding='utf-8') as f:
    app_code = f.read()

target_block = '''        else:
            # Info or general queries
            if 'hello' in message_lower or 'hi ' in message_lower or message_lower == 'hi':
                response = "Hello! I am the CreditScope assistant. Please enter an applicant ID (SK_ID_CURR) to assess their credit risk."
            elif 'help' in message_lower:
                response = "To use this chatbot, simply type or paste an applicant ID (e.g. 100002) from your uploaded dataset. I will run the active models and provide a risk assessment."
            elif 'dataset' in message_lower or 'model' in message_lower:
                models = model_manager.get_models()
                active = sum(1 for m in models if m.get('is_selected', False))
                has_data = current_dataset.get('dataframe') is not None
                data_status = f"Dataset is {'loaded' if has_data else 'NOT loaded'}."
                model_status = f"You have {active} active models out of {len(models)} uploaded."
                response = f"System status: {data_status} {model_status}"
            else:
                response = "I didn't detect an applicant ID in your message. Please provide a valid numerical applicant ID to perform a risk assessment."
                
            return jsonify({
                'type': 'info',
                'response': response,
                'applicant_id': None,
                'applicant_data': None,
                'predictions': None,
                'avg_probability': None,
                'risk_level': None,
                'risk_factors': None
            })'''

new_block = '''        else:
            # Info or general queries - use Gemini if available for conversational AI
            if Config.GEMINI_API_KEY:
                import google.generativeai as genai
                try:
                    genai.configure(api_key=Config.GEMINI_API_KEY)
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    
                    models = model_manager.get_models()
                    active = sum(1 for m in models if m.get('is_selected', False))
                    has_data = current_dataset.get('dataframe') is not None
                    data_status = f"loaded with {len(current_dataset['dataframe'])} rows" if has_data else "NOT loaded"
                    
                    system_prompt = (
                        "You are CreditScope AI, a helpful, professional credit risk analysis assistant. "
                        f"Context: The dataset is currently {data_status}. "
                        f"There are {active} active prediction models out of {len(models)} uploaded. "
                        "If the user wants to assess a specific applicant's risk, instruct them to simply type the applicant's exact ID number (SK_ID_CURR). "
                        "Answer the user's questions clearly, concisely, and conversationally."
                    )
                    
                    gemini_response = model.generate_content([system_prompt, f"User: {message}"])
                    response = gemini_response.text
                except Exception as e:
                    response = f"I encountered an error connecting to my AI brain. (Error: {str(e)})"
            else:
                if 'hello' in message_lower or 'hi ' in message_lower or message_lower == 'hi':
                    response = "Hello! I am the CreditScope assistant. Please enter an applicant ID (SK_ID_CURR) to assess their credit risk."
                elif 'help' in message_lower:
                    response = "To use this chatbot, simply type or paste an applicant ID (e.g. 100002) from your uploaded dataset. I will run the active models and provide a risk assessment."
                else:
                    response = "I didn't detect an applicant ID in your message. Please provide a valid numerical applicant ID to perform a risk assessment."
                    
            return jsonify({
                'type': 'info',
                'response': response,
                'applicant_id': None,
                'applicant_data': None,
                'predictions': None,
                'avg_probability': None,
                'risk_level': None,
                'risk_factors': None
            })'''

app_code = app_code.replace(target_block, new_block)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(app_code)
print('Chatbot updated!')
