import re
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class ChatbotPhongChongLuaDao:
    def __init__(self):
        self.fraud_rules = [
            {
                'pattern': r'yêu.+?gửi.+?tiền|chuyển.+?tiền.+?trước|đặt.+?cọc.+?trước',
                'fraud_type': 'Lừa đảo tài chính',
                'confidence': 0.8,
                'advice': 'Cảnh báo cao: Việc yêu cầu chuyển tiền/đặt cọc trước khi gặp mặt hoặc nhận sản phẩm thường là dấu hiệu của lừa đảo.'
            },
            {
                'pattern': r'trúng.+?thưởng|trúng.+?giải|quà.+?tặng.+?miễn phí',
                'fraud_type': 'Lừa đảo trúng thưởng',
                'confidence': 0.75,
                'advice': 'Cảnh báo: Các thông báo trúng thưởng/quà tặng đột ngột thường là lừa đảo.'
            },
            {
                'pattern': r'đầu tư.+?lãi.+?cao|lợi nhuận.+?khủng|thu nhập.+?thụ động',
                'fraud_type': 'Lừa đảo đầu tư',
                'confidence': 0.85,
                'advice': 'Cảnh báo cao: Các lời mời đầu tư với lãi suất cao bất thường thường là lừa đảo.'
            },
            {
                'pattern': r'cập nhật.+?tài khoản|xác minh.+?thông tin|tài khoản.+?hết hạn',
                'fraud_type': 'Lừa đảo giả mạo (Phishing)',
                'confidence': 0.9,
                'advice': 'Cảnh báo rất cao: Đây là dấu hiệu điển hình của lừa đảo phishing.'
            },
            {
                'pattern': r'việc làm.+?thu nhập cao|làm.+?online.+?lương cao|việc nhẹ lương cao',
                'fraud_type': 'Lừa đảo việc làm',
                'confidence': 0.7,
                'advice': 'Cảnh báo: Các lời mời việc làm với thu nhập cao bất thường mà không yêu cầu kinh nghiệm thường là lừa đảo.'
            },
            {
                'pattern': r'hỗ trợ.+?kỹ thuật|sửa.+?lỗi.+?máy tính|virus.+?cần xử lý',
                'fraud_type': 'Lừa đảo hỗ trợ kỹ thuật',
                'confidence': 0.8,
                'advice': 'Cảnh báo cao: Các cuộc gọi/tin nhắn tự xưng là nhân viên kỹ thuật thường là lừa đảo.'
            },
            {
                'pattern': r'giá rẻ.+?bất thường|deal.+?sock|giảm giá.+?sâu',
                'fraud_type': 'Lừa đảo mua sắm',
                'confidence': 0.6,
                'advice': 'Cảnh báo: Các sản phẩm giá rẻ bất thường có thể là hàng giả, hàng nhái hoặc không tồn tại.'
            },
            {
                'pattern': r'cho vay.+?không cần thế chấp|vay.+?nhanh.+?lãi thấp|vay tiền.+?duyệt ngay',
                'fraud_type': 'Lừa đảo tín dụng',
                'confidence': 0.75,
                'advice': 'Cảnh báo: Các dịch vụ cho vay dễ dàng, không thẩm định thường là lừa đảo hoặc vay nặng lãi.'
            },
            {
                'pattern': r'giúp.+?khẩn cấp|tai nạn.+?cần tiền|bệnh.+?cần hỗ trợ',
                'fraud_type': 'Lừa đảo giả mạo người thân',
                'confidence': 0.85,
                'advice': 'Cảnh báo cao: Đây thường là trò lừa đảo mạo danh người thân/bạn bè gặp hoạn nạn.'
            },
            {
                'pattern': r'đăng ký.+?gấp|số lượng.+?có hạn|chỉ còn.+?suất',
                'fraud_type': 'Lừa đảo tạo tâm lý hối thúc',
                'confidence': 0.65,
                'advice': 'Cảnh báo: Các thông báo tạo cảm giác khẩn cấp, hối thúc quyết định nhanh thường nhằm ngăn bạn suy nghĩ kỹ.'
            }
        ]

        self.X_train = [
            "Chúc mừng! Bạn đã trúng iPhone 15 Pro Max. Vui lòng chuyển phí vận chuyển.",
            "Tài khoản ngân hàng của bạn có vấn đề. Vui lòng cập nhật thông tin.",
            "Đầu tư 10 triệu, thu về 50 triệu sau 1 tháng.",
            "Bạn nhận quà tặng đặc biệt. Cung cấp thông tin cá nhân.",
            "Máy tính có virus. Cho phép truy cập để xử lý.",
            "Việc làm lương cao tại nhà, không cần kinh nghiệm.",
            "Tài khoản Facebook bị khóa. Xác minh lại thông tin.",
            "Chuyển tiền đặt cọc để nhận việc làm.",
            "Gửi ảnh CMND để xác minh danh tính.",
            "Con gặp tai nạn cần tiền gấp.",
            "Vay tiền không cần thế chấp, giải ngân nhanh.",
            "iPhone 14 Pro chỉ còn 5 triệu, số lượng có hạn.",
            "Bạn trúng thưởng quay số. Liên hệ nhận thưởng.",
            "Đầu tư bitcoin, cam kết lợi nhuận 200%.",
            "Ngân hàng thông báo tài khoản bị xâm nhập.",

            # Không phải lừa đảo
            "Cảm ơn bạn đã mua hàng. Đơn đang được xử lý.",
            "Thông báo bảo trì hệ thống vào ngày mai.",
            "Mời bạn tham dự hội thảo bảo mật.",
            "Sản phẩm bạn đặt mua hiện hết hàng.",
            "Xin chào, sản phẩm còn hàng không?",
            "Thông báo lịch nghỉ lễ 30/4 - 1/5.",
            "Chúng tôi đã gửi hóa đơn điện tử qua email.",
            "Bạn có thể thanh toán khi nhận hàng.",
            "Đơn hàng đang giao đến bạn.",
            "Vui lòng đánh giá dịch vụ sau khi nhận hàng."
        ]

        self.y_train = [1]*15 + [0]*10

        self.classifier = Pipeline([
            ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), max_features=1000)),
            ('classifier', MultinomialNB())
        ])

        self.classifier.fit(self.X_train, self.y_train)

        self.fuzzy_risk_levels = {
            'low': lambda x: max(0, min(1, (0.4 - x) / 0.4)) if x <= 0.4 else 0,
            'medium': lambda x: max(0, min((x - 0.2) / 0.3, (0.7 - x) / 0.3)) if 0.2 <= x <= 0.7 else 0,
            'high': lambda x: max(0, min(1, (x - 0.5) / 0.5)) if x >= 0.5 else 0
        }

    def fuzzy_risk_evaluation(self, probability):
        membership_values = {
            level: func(probability) for level, func in self.fuzzy_risk_levels.items()
        }
        max_level = max(membership_values.items(), key=lambda x: x[1])
        return max_level[0] if max_level[1] > 0 else "không xác định"

    def rule_based_detection(self, text):
        for rule in self.fraud_rules:
            if re.search(rule['pattern'], text.lower()):
                return rule
        return None

    def bayes_detection(self, text):
        return self.classifier.predict_proba([text])[0][1]

    def respond(self, user_input):
        rule_result = self.rule_based_detection(user_input)
        bayes_prob = self.bayes_detection(user_input)
        risk_level = self.fuzzy_risk_evaluation(bayes_prob)

        if rule_result:
            response = f"⚠️ <b>CẢNH BÁO: {rule_result['fraud_type']}</b><br><br>"
            response += f"<b>Mức độ rủi ro:</b> {risk_level.upper()} (Độ tin cậy: {int(rule_result['confidence']*100)}%)<br><br>"
            response += f"<b>Lời khuyên:</b> {rule_result['advice']}<br><br>"
            response += f"<i>Phân tích Bayes: Xác suất lừa đảo là {bayes_prob*100:.1f}%</i>"
            return response

        elif bayes_prob > 0.5:
            response = f"⚠️ <b>CẢNH BÁO: Có khả năng là lừa đảo</b><br><br>"
            response += f"<b>Mức độ rủi ro:</b> {risk_level.upper()}<br><br>"
            response += f"<b>Lời khuyên:</b> Nội dung này có dấu hiệu đáng nghi. Hãy cẩn trọng!<br><br>"
            response += f"<i>Phân tích Bayes: Xác suất lừa đảo là {bayes_prob*100:.1f}%</i>"
            return response

        else:
            response = f"✅ <b>Không phát hiện dấu hiệu rõ ràng của lừa đảo</b><br><br>"
            response += f"<b>Mức độ rủi ro:</b> {risk_level}<br><br>"
            response += f"<i>Xác suất lừa đảo: {bayes_prob*100:.1f}%</i><br>"
            response += "Tuy nhiên, hãy luôn cảnh giác với các thông tin bất ngờ hoặc yêu cầu nhạy cảm."
            return response
