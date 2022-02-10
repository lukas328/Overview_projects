//
//  HapticFeedback.swift
//  Meet Me
//
//  Created by Philipp Hemkemeyer on 17.01.21.
//

import SwiftUI

// haptic feedback depending on an action (eg. success, error or warning)
func hapticFeedback(feedBackstyle: UINotificationFeedbackGenerator.FeedbackType) {
    UINotificationFeedbackGenerator().notificationOccurred(feedBackstyle)
}

// haptic pulse can take different strengths 
func hapticPulse(feedback: UIImpactFeedbackGenerator.FeedbackStyle) {
    UIImpactFeedbackGenerator(style: feedback).impactOccurred()
}
