import 'package:flutter_test/flutter_test.dart';
import 'package:lung_pathology_ai/main.dart';

void main() {
  testWidgets('App smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const LungPathologyApp());
    expect(find.text('AI Chest Disease Detection'), findsOneWidget);
  });
}
