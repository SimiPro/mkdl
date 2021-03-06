luanet.load_assembly "System"

local server = {}

---- TCP CONFIGURATION
local Encoder = luanet.import_type "System.Text.Encoding"
local TcpClient = luanet.import_type("System.Net.Sockets.TcpClient")
local IO = luanet.import_type "System.IO.Path"
server.USE_CLIPBOARD = true
server.SCREENSHOT_FILE = ""
server.start_time = os.clock()
server.old_progress = 0
server.new_progress = 0
server.old_velocity = 0
server.tmp_progress = 0
server.old_time = 0
server.new_time = 0
server.jump = false
server.finish_message = false
server.max_time = 100

function server.init(port, use_clipboard, file_name)
  server.mySocket = TcpClient("localhost", port)
  server.stream = server.mySocket:GetStream()
  server.sMessage = "myMessage"
  server.rMessage = "receivedData"
  server.steering_action = 0
  server.USE_CLIPBOARD = use_clipboard
  server.SCREENSHOT_FILE = server.getTMPDir() .. file_name
end

function server.sendMsg()
  local message = Encoder.UTF8:GetBytes(server.sMessage)
  server.stream:Write(message, 0, string.len(server.sMessage))
  --console.log("sent: " .. sMessage)
end

function server.recvData()
  local buffer = "000000000000000000000000000000000" -- we should actually just receive a value to steer
  local byteBuffer = Encoder.UTF8:GetBytes(buffer)
  local bytesSent = server.stream:Read(byteBuffer, 0, string.len(buffer))
  local receivedMessage = Encoder.UTF8:GetString(byteBuffer)
  return string.sub(receivedMessage, 1, bytesSent)
end

function server.getTMPDir()
  return IO.GetTempPath()
end

BOX_CENTER_X, BOX_CENTER_Y = 160, 215
BOX_WIDTH, BOX_HEIGHT = 100, 4
SLIDER_WIDTH, SLIDER_HIEGHT = 4, 16
function server.draw_info()
  gui.drawBox(BOX_CENTER_X - BOX_WIDTH / 2, BOX_CENTER_Y - BOX_HEIGHT / 2,
              BOX_CENTER_X + BOX_WIDTH / 2, BOX_CENTER_Y + BOX_HEIGHT / 2,
              none, 0x60FFFFFF)
  gui.drawBox(BOX_CENTER_X + server.steering_action*(BOX_WIDTH / 2) - SLIDER_WIDTH / 2, BOX_CENTER_Y - SLIDER_HIEGHT / 2,
              BOX_CENTER_X + server.steering_action*(BOX_WIDTH / 2) + SLIDER_WIDTH / 2, BOX_CENTER_Y + SLIDER_HIEGHT / 2,
              none, 0xFFFF0000)
end

---- END TCP CONFIGURATION

---- SOME OTHER GLOBALS

--[[ How many frames to wait before sending a new prediction request. If you're using a file, you
may want to consider adding some frames here. ]]--
local WAIT_FRAMES = 3

savestate.loadslot(2)
savestate.saveslot(2) -- save current slot for reset purposes
local util = require("util")


--- reinforcement variables
local done = "False"
local totalReward = 0


function server.get_reward4()
  local velocity = util.readVelocity()

  return velocity*server.new_progress
end

function server.get_reward3()
  local velocity = util.readVelocity()
  velocity = velocity / 10
  if velocity < .1 then
    velocity = 0
  end
  --gui.addmessage("old: " .. server.old_progress .. " | new: "  .. server.new_progress)
  if server.old_progress > server.new_progress then
    return (-1)*util.round(velocity, 1)
  end
  return util.round(velocity, 1)
end

--- this function gives negativ reward if we slow down
function server.get_reward2()
  if server.old_progress > server.new_progress then
    return -.1
  end

  local new_velocity = util.readVelocity()
  if new_velocity < 5.4 and server.jump then
    return .0
  end
  gui.addmessage(new_velocity)
  if new_velocity > 5.4 then
    return .3
  end
  if new_velocity > 5 then
    return .2
  end
  if new_velocity > 3 then
    return .1
  end
  if new_velocity > server.old_velocity then
    return .0
  elseif new_velocity == server.old_velocity then
    return .0
  else
    return -.1
  end
  server.old_velocity = new_velocity
  return new_velocity
end

function server.create_message()
  local reward = server.get_reward2()
  server.new_time = util.readTimer()
  gui.addmessage("reward: " .. reward)
  totalReward = totalReward + reward
  if server.new_time > server.max_time or totalReward < -3 or util.readProgress() >= 3.1 then -- we reset after 150 seconds
    done = "True"
    totalReward = 0
    console.log(server.new_time)
    server.finish_message = false
  else
    done = "False"
  end
  --console.log(totalReward)

  --console.log(reward)
  if server.USE_CLIPBOARD then
    client.screenshottoclipboard()
    server.sMessage = "MESSAGE screenshot_clip_reward_" .. reward .. "_done_" .. done .. "\n"
  else
    client.screenshot(server.SCREENSHOT_FILE)
    server.sMessage = "MESSAGE screenshot_" .. server.SCREENSHOT_FILE .. "_reward_" .. reward .. "_done_" .. done .. "\n"
    --outgoing_message = "PREDICT:" .. SCREENSHOT_FILE .. "\n"
  end
end

function server.start()
    while util.readProgress() < 3.5 do -- 3 means 3 laps
      if util.readProgress() > 3 and not server.finish_message then
        console.log(util.readTimer())
        server.finish_message = true
      end
      -- Process the outgoing message.
      server.new_progress = util.readProgress()
      server.create_message()
      server.sendMsg()
      --- Process incoming message
      server.rMessage = server.recvData()
      if string.find(server.rMessage, "RESET") ~= nil then
        server.start_time = os.clock()
        console.log('Reset game - LOADING SLOT 2 Which we saved at the beginning')
        savestate.loadslot(2)
        client.unpause()
      elseif string.find(server.rMessage, "PREDICTIONERROR") == nil then
        --console.log('current message: ' .. rMessage)
        local a1, a2 = string.match(server.rMessage, "(-?%d.%d*):(%d*)")
        --console.log('a1: ' .. a1 .. ' a2: ' .. a2)

        if a1 == nil then
          a1 = server.rMessage -- if we only have 1 action
        else
          -- we have 2 actions
          if 1 == tonumber(a2) then
            server.jump = true
            gui.addmessage('jump')
          else
            server.jump = false
          end
        end
        server.steering_action = tonumber(a1)

        --console.log('current message: ' .. rMessage)
        --console.log('current steering_action: ' .. steering_action)

        for i=1, WAIT_FRAMES do
          --console.log('wait frame')
          joypad.setanalog({["P1 X Axis"] = util.convertSteerToJoystick(server.steering_action) })
          joypad.set({["P1 R"] = server.jump, ["P1 A"] = true})
          server.draw_info()
          emu.frameadvance()
        end
      else
        print("Prediction error...")
      end

      server.old_progress = server.new_progress
    end

end

return server

