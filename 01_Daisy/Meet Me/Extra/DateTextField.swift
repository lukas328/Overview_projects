//
//  DateField.swift
//
//  Created by Szymon Lorenz on 22/2/20.
//  Modified by Philipp Hemkemeyer on 07/02/21
//  Copyright © 2020 Szymon Lorenz. All rights reserved.
//

import SwiftUI

struct DateTextField: UIViewRepresentable {
    @Binding var date: Date
    
    // format to show the dates
    var localeStyle: String = "en_DE"
    public var dateFormat: String = "dd/MM/YY"
    
    var didChange: () -> Void = { }
    
    private var minimumDate: Date? = Date()
    private var maximumDate: Date? = nil
    private var placeholder: String? = "Select a date"
    private var dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .long
        return formatter
    }()
    
    private var font: UIFont?
    private var foregroundColor: UIColor?
    private var accentColor: UIColor?
    private var textAlignment: NSTextAlignment?
    private var contentType: UITextContentType?
    
    private var autocorrection: UITextAutocorrectionType = .default
    private var autocapitalization: UITextAutocapitalizationType = .sentences
    private var keyboardType: UIKeyboardType = .default
    private var returnKeyType: UIReturnKeyType = .default
    
    private var isSecure: Bool = false
    private var isUserInteractionEnabled: Bool = true
    private var clearsOnBeginEditing: Bool = false
    
    @Environment(\.layoutDirection) private var layoutDirection: LayoutDirection
    
    init(date: Binding<Date>,
         didChange: @escaping () -> Void = { })
    {
        self._date = date
        self.didChange = didChange
    }
    
    func makeUIView(context: Context) -> UITextField {
        let textField = UITextField()
        
        textField.delegate = context.coordinator
        
        // German format DE and English language en
        dateFormatter.locale = Locale(identifier: localeStyle)
        textField.text = dateFormatter.string(from: date)
        textField.font = font
        textField.textColor = foregroundColor
        if let textAlignment = textAlignment {
            textField.textAlignment = textAlignment
        }
        if let contentType = contentType {
            textField.textContentType = contentType
        }
        if let accentColor = accentColor {
            textField.tintColor = accentColor
        }
        textField.autocorrectionType = autocorrection
        textField.autocapitalizationType = autocapitalization
        textField.keyboardType = keyboardType
        textField.returnKeyType = returnKeyType
        
        textField.clearsOnBeginEditing = clearsOnBeginEditing
        textField.isSecureTextEntry = isSecure
        textField.isUserInteractionEnabled = isUserInteractionEnabled
        
        textField.setContentHuggingPriority(.defaultHigh, for: .vertical)
        textField.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        
        
        let datePickerView = UIDatePicker()
        datePickerView.datePickerMode = .date
        datePickerView.maximumDate = minimumDate
        datePickerView.maximumDate = maximumDate
        
        
        
        
        // setting the design to the old look which is inside the keyboard area
        datePickerView.preferredDatePickerStyle = .wheels
        datePickerView.locale = Locale(identifier: localeStyle)
        // if not the standard view then use only the time
        if dateFormat != "dd/MM/YY" {
            datePickerView.datePickerMode = .time
        }
        textField.inputView = datePickerView
        datePickerView.addTarget(context.coordinator, action: #selector(Coordinator.handleDatePicker(sender:)), for: .valueChanged)
        addDoneButtonToKeyboard(textField)
        return textField
    }
    
    func updateUIView(_ uiView: UITextField, context: Context) {
        // German format DE and English language en
        dateFormatter.locale = Locale(identifier: localeStyle)
        dateFormatter.dateFormat = dateFormat
        uiView.text = dateFormatter.string(from: date)
    }
    
    private func addDoneButtonToKeyboard(_ view: UITextField) {
        let doneToolbar: UIToolbar = UIToolbar(frame: CGRect(x: 0, y: 0, width: UIScreen.main.bounds.width, height: 35))
        doneToolbar.barStyle       = .default
        let flexSpace              = UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: nil, action: nil)
        let done: UIBarButtonItem  = UIBarButtonItem(title: "Done", style: .done, target: view, action: #selector(UITextField.resignFirstResponder))
        done.setTitleTextAttributes([NSAttributedString.Key.font:UIFont.boldSystemFont(ofSize: 18)], for: .normal)
        
        var items = [UIBarButtonItem]()
        items.append(flexSpace)
        items.append(done)
        
        doneToolbar.items = items
        doneToolbar.sizeToFit()
        
        view.inputAccessoryView = doneToolbar
    }
    
    func makeCoordinator() -> Coordinator {
        return Coordinator(date: $date,
                           didChange: didChange)
    }
    
    final class Coordinator: NSObject, UITextFieldDelegate {
        @Binding var date: Date
        var didChange: () -> Void
        
        init(date: Binding<Date>, didChange: @escaping () -> Void) {
            self._date = date
            self.didChange = didChange
        }
        
        @objc func handleDatePicker(sender: UIDatePicker) {
            date = sender.date
            didChange()
        }
        
        func textFieldShouldReturn(_ textField: UITextField) -> Bool {
            return false
        }
    }
}


extension DateTextField {
    func minimumDate(_ date: Date?) -> some View {
        var view = self
        view.minimumDate = date
        return view
    }
    
    func maximumDate(_ date: Date?) -> some View {
        var view = self
        view.maximumDate = date
        return view
    }
    
    func placeholder(_ text: String?) -> some View {
        var view = self
        view.placeholder = text
        return view
    }
    
    // func to only change the dateFormat of the picker
    func dateFormat(_ dateFormat: String) -> some View {
        var view = self
        view.dateFormat = dateFormat
        return view
    }
    
    func dateFormatter(_ formatter: DateFormatter) -> some View {
        var view = self
        view.dateFormatter = formatter
        return view
    }
    
    func font(_ font: UIFont?) -> some View {
        var view = self
        view.font = font
        return view
    }
    
    func foregroundColor(_ color: UIColor?) -> some View {
        var view = self
        view.foregroundColor = color
        return view
    }
    
    func accentColor(_ accentColor: UIColor?) -> some View {
        var view = self
        view.accentColor = accentColor
        return view
    }
    
    func multilineTextAlignment(_ alignment: TextAlignment) -> some View {
        var view = self
        switch alignment {
        case .leading:
            view.textAlignment = layoutDirection ~= .leftToRight ? .left : .right
        case .trailing:
            view.textAlignment = layoutDirection ~= .leftToRight ? .right : .left
        case .center:
            view.textAlignment = .center
        }
        return view
    }
    
    func textContentType(_ textContentType: UITextContentType?) -> some View {
        var view = self
        view.contentType = textContentType
        return view
    }
    
    func disableAutocorrection(_ disable: Bool?) -> some View {
        var view = self
        if let disable = disable {
            view.autocorrection = disable ? .no : .yes
        } else {
            view.autocorrection = .default
        }
        return view
    }
    
    func autocapitalization(_ style: UITextAutocapitalizationType) -> some View {
        var view = self
        view.autocapitalization = style
        return view
    }
    
    func keyboardType(_ type: UIKeyboardType) -> some View {
        var view = self
        view.keyboardType = type
        return view
    }
    
    func returnKeyType(_ type: UIReturnKeyType) -> some View {
        var view = self
        view.returnKeyType = type
        return view
    }
    
    func isSecure(_ isSecure: Bool) -> some View {
        var view = self
        view.isSecure = isSecure
        return view
    }
    
    func clearsOnBeginEditing(_ shouldClear: Bool) -> some View {
        var view = self
        view.clearsOnBeginEditing = shouldClear
        return view
    }
    
    func disabled(_ disabled: Bool) -> some View {
        var view = self
        view.isUserInteractionEnabled = disabled
        return view
    }
}
