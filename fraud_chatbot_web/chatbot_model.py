
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class ChatbotPhongChongLuaDao:
    def __init__(self):
        from fraud_rules import fraud_rules
        from train_data_augmented import X_train, y_train
        self.fraud_rules = fraud_rules
        self.X_train = X_train
        self.y_train = y_train

        self.classifier = Pipeline([
            ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), strip_accents='unicode')),
            ('classifier', MultinomialNB())
        ])
        self.classifier.fit(self.X_train, self.y_train)

        self.fuzzy_risk_levels = {
            'low': lambda x: max(0, min(1, (0.4 - x) / 0.4)) if x <= 0.4 else 0,
            'medium': lambda x: max(0, min((x - 0.2) / 0.3, (0.7 - x) / 0.3)) if 0.2 <= x <= 0.7 else 0,
            'high': lambda x: max(0, min(1, (x - 0.5) / 0.5)) if x >= 0.5 else 0
        }

    def normalize_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def map_label_to_text(self, label):
        mapping = {
            'trung_thuong': 'Lừa đảo trúng thưởng',
            'phishing': 'Lừa đảo giả mạo (Phishing)',
            'dau_tu': 'Lừa đảo đầu tư',
            'tai_chinh': 'Lừa đảo tài chính',
            'mua_sam': 'Lừa đảo mua sắm',
            'viec_lam': 'Lừa đảo việc làm',
            'ho_tro': 'Lừa đảo hỗ trợ kỹ thuật',
            'gia_mao_nguoi_than': 'Lừa đảo giả mạo người thân',
            'tin_dung': 'Lừa đảo tín dụng',
            'hop_le': 'Nội dung bình thường không có dấu hiệu lừa đảo'
        }
        return mapping.get(label, label)

    def fuzzy_risk_evaluation(self, probability):
        membership_values = {level: func(probability) for level, func in self.fuzzy_risk_levels.items()}
        max_level = max(membership_values.items(), key=lambda x: x[1])
        return max_level[0] if max_level[1] > 0 else "không xác định"

    def rule_based_detection(self, text):
        cleaned = self.normalize_text(text)
        matched_rules = []
        for rule in self.fraud_rules:
            if 'pattern' in rule and re.search(rule['pattern'], cleaned):
                matched_rules.append(rule)
            elif 'keywords' in rule and any(kw in cleaned for kw in rule['keywords']):
                matched_rules.append(rule)
        return matched_rules

    def bayes_detection_detail(self, text):
        proba = self.classifier.predict_proba([text])[0]
        labels = self.classifier.classes_
        return sorted(zip(labels, proba), key=lambda x: x[1], reverse=True)

    def respond(self, user_input):
        rules = self.rule_based_detection(user_input)
        bayes_results = self.bayes_detection_detail(user_input)
        top_label, top_prob = bayes_results[0]
        risk_level = self.fuzzy_risk_evaluation(top_prob)

        response = []
        response.append("🛡️========== KẾT QUẢ PHÂN TÍCH ==========🛡️")

        if rules:
            response.append("📏 [Phát hiện theo luật]")

            # Sắp xếp tất cả luật theo độ tin cậy giảm dần
            sorted_rules = sorted(rules, key=lambda r: r['confidence'], reverse=True)

            # Duyệt và chọn tối đa 3 loại khác nhau
            seen_types = set()
            unique_rules = []
            for rule in sorted_rules:
                if rule['fraud_type'] not in seen_types:
                    unique_rules.append(rule)
                    seen_types.add(rule['fraud_type'])
                if len(unique_rules) == 3:
                    break

            # In ra kết quả
            for rule in unique_rules:
                response.append(f"• Loại: {rule['fraud_type']}")
                response.append(f"  ➤ Khuyến nghị: {rule['advice']}")
            response.append("")

        # Phân tích mô hình
        response.append("🧠 [Phân tích từ mô hình học máy]")
        for i, (label, prob) in enumerate(bayes_results[:5], 1):
            response.append(f"{i}. {self.map_label_to_text(label)} — {prob*100:.2f}%")

        # Mức rủi ro tổng thể
        response.append("")
        response.append(f"🚨 [Mức độ rủi ro tổng thể]: {risk_level.upper()}")

        # Gợi ý nếu rủi ro cao
        if top_label != "hop_le" and top_prob >= 0.6:
            response.append("")
            response.append("💡 [Cảnh báo]")
            response.append("Nội dung có dấu hiệu RỦI RO CAO.")
            response.append("❗ Hãy xác minh nguồn gốc trước khi cung cấp thông tin hoặc chuyển tiền.")

        response.append("🛡️=======================================🛡️")
        return "\n".join(response)
