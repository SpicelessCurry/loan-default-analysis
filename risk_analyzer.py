import numpy as np

class RiskAnalyzer:
    FEATURE_CATEGORIES = {
        'Financial Profile': ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'CREDIT_INCOME_RATIO', 'INCOME_GROUP', 'CREDIT_GROUP'],
        'Bureau Credit History': ['BUREAU_CREDIT_COUNT', 'BUREAU_ACTIVE_COUNT', 'BUREAU_ACTIVE_RATIO', 'BUREAU_OVERDUE_COUNT', 'BUREAU_OVERDUE_RATIO', 'BUREAU_MAX_OVERDUE_DAYS', 'HAS_BUREAU_HISTORY', 'HAS_BUREAU_OVERDUE', 'OVERDUE_SEVERITY', 'BUREAU_TOTAL_DEBT', 'BUREAU_DEBT_INCOME_RATIO'],
        'Bureau Balance History': ['BUREAU_BALANCE_CREDIT_COUNT', 'BUREAU_TOTAL_OVERDUE_MONTHS', 'BUREAU_TOTAL_SEVERE_OVERDUE_MONTHS', 'BUREAU_MAX_OVERDUE_RATIO', 'HAS_BUREAU_BALANCE_HISTORY'],
        'Previous Applications': ['PREVIOUS_APP_COUNT', 'PREVIOUS_REFUSED_COUNT', 'PREVIOUS_CANCELED_COUNT', 'PREVIOUS_AVG_CREDIT', 'PREVIOUS_APPROVAL_RATE', 'PREVIOUS_REFUSAL_RATE'],
        'Payment Behavior': ['LATE_PAYMENT_COUNT', 'LATE_PAYMENT_RATE', 'AVG_PAYMENT_DELAY', 'UNDERPAYMENT_RATE'],
        'Demographics': ['AGE_YEARS', 'AGE_GROUP', 'CODE_GENDER', 'CNT_CHILDREN', 'NAME_EDUCATION_TYPE'],
        'Employment & Income': ['DAYS_EMPLOYED', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'NAME_INCOME_TYPE_Working', 'NAME_INCOME_TYPE_Pensioner', 'NAME_INCOME_TYPE_State servant'],
        'External Sources': ['EXT_SOURCE_2', 'EXT_SOURCE_3'],
        'Property & Housing': ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'YEARS_BEGINEXPLUATATION_MEDI', 'FLOORSMAX_MEDI', 'TOTALAREA_MODE'],
        'Social Circle': ['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE'],
        'Credit Bureau Inquiries': ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
    }
    
    def _is_valid_value(self, data, key):
        if key not in data:
            return False
        val = data[key]
        if val is None:
            return False
        if isinstance(val, float) and np.isnan(val):
            return False
        return True

    def analyze_risk_factors(self, applicant_data: dict) -> list:
        risk_factors = []
        d = applicant_data
        
        # Financial Profile
        if self._is_valid_value(d, 'CREDIT_INCOME_RATIO'):
            if d['CREDIT_INCOME_RATIO'] > 5:
                risk_factors.append({'feature': 'CREDIT_INCOME_RATIO', 'value': d['CREDIT_INCOME_RATIO'], 'category': 'Financial Profile', 'severity': 'high', 'description': 'Credit amount is very high relative to income'})
            elif d['CREDIT_INCOME_RATIO'] > 3:
                risk_factors.append({'feature': 'CREDIT_INCOME_RATIO', 'value': d['CREDIT_INCOME_RATIO'], 'category': 'Financial Profile', 'severity': 'medium', 'description': 'Credit to income ratio is elevated'})
        
        if self._is_valid_value(d, 'AMT_INCOME_TOTAL') and d['AMT_INCOME_TOTAL'] < 100000:
            risk_factors.append({'feature': 'AMT_INCOME_TOTAL', 'value': d['AMT_INCOME_TOTAL'], 'category': 'Financial Profile', 'severity': 'medium', 'description': 'Low income level'})
            
        if self._is_valid_value(d, 'AMT_CREDIT') and d['AMT_CREDIT'] > 1000000:
            risk_factors.append({'feature': 'AMT_CREDIT', 'value': d['AMT_CREDIT'], 'category': 'Financial Profile', 'severity': 'medium', 'description': 'High credit amount'})
            
        # Bureau History
        if self._is_valid_value(d, 'BUREAU_OVERDUE_COUNT') and d['BUREAU_OVERDUE_COUNT'] > 0:
            risk_factors.append({'feature': 'BUREAU_OVERDUE_COUNT', 'value': d['BUREAU_OVERDUE_COUNT'], 'category': 'Bureau Credit History', 'severity': 'high', 'description': 'Has overdue payments in credit bureau'})
            
        if self._is_valid_value(d, 'BUREAU_OVERDUE_RATIO') and d['BUREAU_OVERDUE_RATIO'] > 0.3:
            risk_factors.append({'feature': 'BUREAU_OVERDUE_RATIO', 'value': d['BUREAU_OVERDUE_RATIO'], 'category': 'Bureau Credit History', 'severity': 'high', 'description': 'High ratio of overdue accounts'})
            
        if self._is_valid_value(d, 'BUREAU_MAX_OVERDUE_DAYS') and d['BUREAU_MAX_OVERDUE_DAYS'] > 60:
            risk_factors.append({'feature': 'BUREAU_MAX_OVERDUE_DAYS', 'value': d['BUREAU_MAX_OVERDUE_DAYS'], 'category': 'Bureau Credit History', 'severity': 'high', 'description': 'Significant overdue history'})
            
        if self._is_valid_value(d, 'BUREAU_DEBT_INCOME_RATIO') and d['BUREAU_DEBT_INCOME_RATIO'] > 1:
            risk_factors.append({'feature': 'BUREAU_DEBT_INCOME_RATIO', 'value': d['BUREAU_DEBT_INCOME_RATIO'], 'category': 'Bureau Credit History', 'severity': 'high', 'description': 'Bureau debt exceeds income'})
            
        if self._is_valid_value(d, 'HAS_BUREAU_OVERDUE') and d['HAS_BUREAU_OVERDUE'] == 1:
            risk_factors.append({'feature': 'HAS_BUREAU_OVERDUE', 'value': d['HAS_BUREAU_OVERDUE'], 'category': 'Bureau Credit History', 'severity': 'medium', 'description': 'Has bureau overdue history'})
            
        if self._is_valid_value(d, 'BUREAU_TOTAL_DEBT') and d['BUREAU_TOTAL_DEBT'] > 500000:
            risk_factors.append({'feature': 'BUREAU_TOTAL_DEBT', 'value': d['BUREAU_TOTAL_DEBT'], 'category': 'Bureau Credit History', 'severity': 'medium', 'description': 'High total bureau debt'})

        # Bureau Balance
        if self._is_valid_value(d, 'BUREAU_TOTAL_OVERDUE_MONTHS') and d['BUREAU_TOTAL_OVERDUE_MONTHS'] > 3:
            risk_factors.append({'feature': 'BUREAU_TOTAL_OVERDUE_MONTHS', 'value': d['BUREAU_TOTAL_OVERDUE_MONTHS'], 'category': 'Bureau Balance History', 'severity': 'high', 'description': 'Multiple months of overdue payments'})
            
        if self._is_valid_value(d, 'BUREAU_TOTAL_SEVERE_OVERDUE_MONTHS') and d['BUREAU_TOTAL_SEVERE_OVERDUE_MONTHS'] > 0:
            risk_factors.append({'feature': 'BUREAU_TOTAL_SEVERE_OVERDUE_MONTHS', 'value': d['BUREAU_TOTAL_SEVERE_OVERDUE_MONTHS'], 'category': 'Bureau Balance History', 'severity': 'high', 'description': 'Has severe overdue months'})
            
        if self._is_valid_value(d, 'BUREAU_MAX_OVERDUE_RATIO') and d['BUREAU_MAX_OVERDUE_RATIO'] > 0.5:
            risk_factors.append({'feature': 'BUREAU_MAX_OVERDUE_RATIO', 'value': d['BUREAU_MAX_OVERDUE_RATIO'], 'category': 'Bureau Balance History', 'severity': 'high', 'description': 'High overdue ratio in bureau balance'})
            
        # Previous Applications
        if self._is_valid_value(d, 'PREVIOUS_REFUSAL_RATE') and d['PREVIOUS_REFUSAL_RATE'] > 0.5:
            risk_factors.append({'feature': 'PREVIOUS_REFUSAL_RATE', 'value': d['PREVIOUS_REFUSAL_RATE'], 'category': 'Previous Applications', 'severity': 'high', 'description': 'High rate of previous loan refusals'})
            
        if self._is_valid_value(d, 'PREVIOUS_REFUSED_COUNT') and d['PREVIOUS_REFUSED_COUNT'] > 3:
            risk_factors.append({'feature': 'PREVIOUS_REFUSED_COUNT', 'value': d['PREVIOUS_REFUSED_COUNT'], 'category': 'Previous Applications', 'severity': 'medium', 'description': 'Multiple previous loan refusals'})
            
        if self._is_valid_value(d, 'PREVIOUS_APPROVAL_RATE') and self._is_valid_value(d, 'PREVIOUS_APP_COUNT'):
            if d['PREVIOUS_APPROVAL_RATE'] < 0.3 and d['PREVIOUS_APP_COUNT'] > 2:
                risk_factors.append({'feature': 'PREVIOUS_APPROVAL_RATE', 'value': d['PREVIOUS_APPROVAL_RATE'], 'category': 'Previous Applications', 'severity': 'medium', 'description': 'Low approval rate'})
                
        # Payment Behavior
        if self._is_valid_value(d, 'LATE_PAYMENT_RATE') and d['LATE_PAYMENT_RATE'] > 0.3:
            risk_factors.append({'feature': 'LATE_PAYMENT_RATE', 'value': d['LATE_PAYMENT_RATE'], 'category': 'Payment Behavior', 'severity': 'high', 'description': 'Frequent late payments'})
            
        if self._is_valid_value(d, 'AVG_PAYMENT_DELAY') and d['AVG_PAYMENT_DELAY'] > 30:
            risk_factors.append({'feature': 'AVG_PAYMENT_DELAY', 'value': d['AVG_PAYMENT_DELAY'], 'category': 'Payment Behavior', 'severity': 'high', 'description': 'Significant average payment delay'})
            
        if self._is_valid_value(d, 'UNDERPAYMENT_RATE') and d['UNDERPAYMENT_RATE'] > 0.3:
            risk_factors.append({'feature': 'UNDERPAYMENT_RATE', 'value': d['UNDERPAYMENT_RATE'], 'category': 'Payment Behavior', 'severity': 'high', 'description': 'Frequent underpayments'})
            
        if self._is_valid_value(d, 'LATE_PAYMENT_COUNT') and d['LATE_PAYMENT_COUNT'] > 5:
            risk_factors.append({'feature': 'LATE_PAYMENT_COUNT', 'value': d['LATE_PAYMENT_COUNT'], 'category': 'Payment Behavior', 'severity': 'medium', 'description': 'Multiple late payments'})

        # Demographics
        if self._is_valid_value(d, 'AGE_YEARS') and d['AGE_YEARS'] < 25:
            risk_factors.append({'feature': 'AGE_YEARS', 'value': d['AGE_YEARS'], 'category': 'Demographics', 'severity': 'low', 'description': 'Young applicant with limited credit history'})
            
        if self._is_valid_value(d, 'CNT_CHILDREN') and d['CNT_CHILDREN'] > 4:
            risk_factors.append({'feature': 'CNT_CHILDREN', 'value': d['CNT_CHILDREN'], 'category': 'Demographics', 'severity': 'low', 'description': 'Large number of dependents'})
            
        # External Sources
        if self._is_valid_value(d, 'EXT_SOURCE_2'):
            if d['EXT_SOURCE_2'] < 0.3:
                risk_factors.append({'feature': 'EXT_SOURCE_2', 'value': d['EXT_SOURCE_2'], 'category': 'External Sources', 'severity': 'high', 'description': 'Low external credit score (Source 2)'})
            elif d['EXT_SOURCE_2'] < 0.5:
                risk_factors.append({'feature': 'EXT_SOURCE_2', 'value': d['EXT_SOURCE_2'], 'category': 'External Sources', 'severity': 'medium', 'description': 'Below average external score (Source 2)'})
                
        if self._is_valid_value(d, 'EXT_SOURCE_3'):
            if d['EXT_SOURCE_3'] < 0.3:
                risk_factors.append({'feature': 'EXT_SOURCE_3', 'value': d['EXT_SOURCE_3'], 'category': 'External Sources', 'severity': 'high', 'description': 'Low external credit score (Source 3)'})
            elif d['EXT_SOURCE_3'] < 0.5:
                risk_factors.append({'feature': 'EXT_SOURCE_3', 'value': d['EXT_SOURCE_3'], 'category': 'External Sources', 'severity': 'medium', 'description': 'Below average external score (Source 3)'})

        # Social Circle
        if self._is_valid_value(d, 'DEF_30_CNT_SOCIAL_CIRCLE') and d['DEF_30_CNT_SOCIAL_CIRCLE'] > 0:
            risk_factors.append({'feature': 'DEF_30_CNT_SOCIAL_CIRCLE', 'value': d['DEF_30_CNT_SOCIAL_CIRCLE'], 'category': 'Social Circle', 'severity': 'medium', 'description': 'Defaults observed in social circle'})
            
        # Employment
        if self._is_valid_value(d, 'DAYS_EMPLOYED'):
            if d['DAYS_EMPLOYED'] > 0:
                risk_factors.append({'feature': 'DAYS_EMPLOYED', 'value': d['DAYS_EMPLOYED'], 'category': 'Employment & Income', 'severity': 'medium', 'description': 'Currently unemployed or irregular employment'})
            elif -365 < d['DAYS_EMPLOYED'] < -182.5:
                 risk_factors.append({'feature': 'DAYS_EMPLOYED', 'value': d['DAYS_EMPLOYED'], 'category': 'Employment & Income', 'severity': 'low', 'description': 'Short employment history'})

        # Property
        if self._is_valid_value(d, 'FLAG_OWN_CAR') and self._is_valid_value(d, 'FLAG_OWN_REALTY'):
            if d['FLAG_OWN_CAR'] == 0 and d['FLAG_OWN_REALTY'] == 0:
                risk_factors.append({'feature': 'FLAG_OWN_CAR', 'value': 0, 'category': 'Property & Housing', 'severity': 'low', 'description': 'No owned assets'})

        # Sort by severity
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        risk_factors.sort(key=lambda x: severity_order.get(x['severity'], 3))
        
        return risk_factors

    def get_risk_level(self, probability: float) -> dict:
        if probability < 0.3:
            return {'level': 'Low Risk', 'color': '#10b981', 'emoji': '🟢', 'description': 'The applicant demonstrates a strong credit profile with minimal risk factors. Approval is generally recommended based on quantitative metrics.'}
        elif probability < 0.5:
            return {'level': 'Moderate Risk', 'color': '#f59e0b', 'emoji': '🟡', 'description': 'The applicant has some risk factors that warrant attention. Standard approval may proceed, but terms might need adjustment or manual review.'}
        elif probability < 0.7:
            return {'level': 'High Risk', 'color': '#f97316', 'emoji': '🟠', 'description': 'The applicant presents significant risk factors. Careful manual review and mitigation strategies (e.g., collateral, higher rates) are strongly advised.'}
        else:
            return {'level': 'Very High Risk', 'color': '#ef4444', 'emoji': '🔴', 'description': 'The applicant exhibits severe risk indicators. Standard approval is not recommended; highly restrictive terms or rejection should be considered.'}

    def generate_rule_based_assessment(self, applicant_data, predictions, risk_factors, risk_level):
        lines = []
        lines.append(f"# Credit Risk Assessment Report")
        lines.append(f"\n## Overall Risk Level: {risk_level['emoji']} {risk_level['level']}")
        lines.append(f"{risk_level['description']}")
        
        lines.append(f"\n## Model Predictions")
        for p in predictions:
            lines.append(f"- **{p['model_name']} ({p['model_type']})**: {p['probability']:.2%} Probability of Default ({p['label']})")
            
        lines.append(f"\n## Key Risk Factors Identified")
        if not risk_factors:
            lines.append("No significant risk factors identified based on the established rules.")
        else:
            # Group by category
            categories = {}
            for rf in risk_factors:
                cat = rf['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(rf)
                
            for cat, factors in categories.items():
                lines.append(f"\n### {cat}")
                for rf in factors:
                    sev = rf['severity'].upper()
                    lines.append(f"- **[{sev}]** {rf['description']} ({rf['feature']}: {rf['value']})")
                    
        lines.append(f"\n## Positive Indicators")
        lines.append("Review standard metrics like income stability and valid contact information to supplement this assessment.")
        
        lines.append(f"\n## Recommendations")
        if risk_level['level'] == 'Low Risk':
            lines.append("Proceed with standard approval process. No special conditions required.")
        elif risk_level['level'] == 'Moderate Risk':
            lines.append("Consider approval with slightly adjusted terms or additional verification steps.")
        elif risk_level['level'] == 'High Risk':
            lines.append("Requires manual underwriting review. Consider requesting collateral or a co-signer.")
        else:
            lines.append("High probability of default. Consider declining the application unless significant mitigants exist.")
            
        return "\n".join(lines)

    def generate_llm_assessment(self, applicant_data, predictions, risk_factors, risk_level, api_key):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            prompt = f"""
            Act as a Senior Credit Risk Analyst. Please write a professional credit risk assessment report based on the following data.
            
            Overall Risk Level: {risk_level['level']} ({risk_level['description']})
            
            Model Predictions for Probability of Default:
            """
            for p in predictions:
                prompt += f"- {p['model_name']} ({p['model_type']}): {p['probability']:.2%}\n"
                
            prompt += "\nIdentified Risk Factors:\n"
            for rf in risk_factors:
                prompt += f"- [{rf['severity'].upper()}] {rf['category']}: {rf['description']} ({rf['feature']} = {rf['value']})\n"
                
            prompt += """
            Write a detailed, structured assessment report. Include:
            1. Executive Summary
            2. Detailed Analysis of Risk Factors
            3. Strengths / Mitigating Factors (extrapolate from the lack of certain risks if necessary)
            4. Final Recommendation (Approve, Reject, or Approve with Conditions)
            
            Use professional financial terminology and keep formatting clean (Markdown).
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Generation failed: {e}")
            return self.generate_rule_based_assessment(applicant_data, predictions, risk_factors, risk_level)

    def generate_assessment(self, applicant_data, predictions, risk_factors, risk_level, api_key=None):
        if api_key:
            return self.generate_llm_assessment(applicant_data, predictions, risk_factors, risk_level, api_key)
        else:
            return self.generate_rule_based_assessment(applicant_data, predictions, risk_factors, risk_level)
