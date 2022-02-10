//
//  PickerTextField.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 27.01.21.
//
import Foundation
import SwiftUI

struct PickerTextField: UIViewRepresentable {
    
    private let textField = UITextField()
    private let pickerView = UIPickerView()
    private let helpForClosingObject = HelpForClosing()
    
    // add the data
    var data: [String]
    var placerholder: String
    
    // index of the selected item
    @Binding var lastSelectedIndex: Int?
    
    func makeUIView(context: Context) -> UITextField {
        self.pickerView.delegate = context.coordinator
        self.pickerView.dataSource = context.coordinator
        
        self.textField.placeholder = self.placerholder
        
        // add the picker to the input
        self.textField.inputView = self.pickerView
        
        
        // configure closing of the keyboard cia accessory view
        let toolbar = UIToolbar(frame: CGRect(x: 0, y: 0, width: UIScreen.main.bounds.width, height: 35))
        toolbar.sizeToFit()
        self.textField.inputAccessoryView = toolbar
        let flexibleSpace = UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: nil, action: nil)
        let doneButton = UIBarButtonItem(title: "Done", style: .plain, target: self.helpForClosingObject, action: #selector(self.helpForClosingObject.doneButtonAction))
        toolbar.setItems([flexibleSpace, doneButton], animated: true)
        
        
        // when the "done" button is tapped
        self.helpForClosingObject.doneButtonTapped = {
            
            // check if index is not nil
            if self.lastSelectedIndex == nil {
                self.lastSelectedIndex = 0
            }
            // hide pickerView via resignFirstResponder
            self.textField.resignFirstResponder()
            
            
        }
        return self.textField
    }
    
    func updateUIView(_ uiView: UIViewType, context: Context) {
        if let lastSelectedIndex = self.lastSelectedIndex {
            uiView.text = self.data[lastSelectedIndex]
        }
    }
    
    func makeCoordinator() -> Coordinator {
        return Coordinator(data: self.data) { (index) in
            self.lastSelectedIndex = index
        }
    }
    
    // this class is need because a @objc method is need and therefore a class instead of a struct is needed
    class HelpForClosing {
        
        public var doneButtonTapped: (() -> Void)?
        
        @objc func doneButtonAction() {
            self.doneButtonTapped?()
        }
    }
    
    class Coordinator: NSObject, UIPickerViewDelegate, UIPickerViewDataSource {
        
        private var data: [String]
        private var didSelectItem: ((Int?) -> Void)?
        
        init(data: [String], didSelectItem: ((Int?) -> Void)? = nil) {
            self.data = data
            self.didSelectItem = didSelectItem
        }
        
        func numberOfComponents(in pickerView: UIPickerView) -> Int {
            return 1
        }
        
        func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
            return self.data.count
        }
        
        func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
            return self.data[row]
        }
        
        func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
            self.didSelectItem?(row)
        }
    }
    
}
