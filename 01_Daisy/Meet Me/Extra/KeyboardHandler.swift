//
//  KeyboardHandler.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 14.02.21.
//

import Combine
import SwiftUI

final class KeyboardHandler: ObservableObject {
    
    // keyboardheight gets update each time the keyboard shows/hides
    @Published private(set) var keyboardHeight: CGFloat = 0
    
    private var cancellable: AnyCancellable?
    
    private let keyboardboardWillShow = NotificationCenter.default
        .publisher(for: UIResponder.keyboardWillShowNotification)
        .compactMap {
            ($0.userInfo?[UIResponder.keyboardFrameEndUserInfoKey] as? CGRect)?.height
        }
    
    private let keyboardboardWillHide = NotificationCenter.default
        .publisher(for: UIResponder.keyboardWillHideNotification)
        .map { _ in CGFloat.zero }
    
    init() {
        cancellable = Publishers.Merge(keyboardboardWillShow, keyboardboardWillHide)
            .subscribe(on: DispatchQueue.main)
            .assign(to: \.self.keyboardHeight, on: self)
    }
    
    func keyboardShown() -> Bool {
        if keyboardHeight != 0 {
            return true
        } else {
            return false
        }
    }
}
